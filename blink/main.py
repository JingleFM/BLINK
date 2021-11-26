# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
from typing import List

import numpy as np
import torch
import uvicorn
from colorama import init
from fastapi import FastAPI
from termcolor import colored
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
import blink.ner as NER
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.biencoder.data_process import (
    get_candidate_representation,
    process_mention_data,
)
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.crossencoder.train_cross import evaluate, modify
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import neuralcoref
import spacy


HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id, e_title, e_text, e_url, show_url=False
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])
    if show_url:
        to_print += "url:{}\n".format(e_url)
    print(to_print)


def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger, context_len, silent=False)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    predictions = np.argsort(logits, axis=1)
    return accuracy, predictions, logits


class EntityLinker:
    def __init__(self, args, logger=None) -> None:
        self.args = args
        self.logger = logger
        # Load NER model
        if logger:
            logger.info("loading NER model")
        with open(args.ner_config) as json_file:
            ner_params = json.load(json_file)
        self.nlp = spacy.load("en_core_web_lg")
        neuralcoref.add_to_pipe(self.nlp)
        ner_params["nlp"] = self.nlp
        self.ner_model = NER.get_model(ner_params)
        # load biencoder model
        if logger:
            logger.info("loading biencoder model")
        with open(args.biencoder_config) as json_file:
            biencoder_params = json.load(json_file)
            biencoder_params["path_to_model"] = args.biencoder_model
        biencoder = load_biencoder(biencoder_params)

        crossencoder = None
        crossencoder_params = None
        if not args.fast:
            # load crossencoder model
            if logger:
                logger.info("loading crossencoder model")
            with open(args.crossencoder_config) as json_file:
                crossencoder_params = json.load(json_file)
                crossencoder_params["path_to_model"] = args.crossencoder_model
            crossencoder = load_crossencoder(crossencoder_params)

        # load candidate entities
        if logger:
            logger.info("loading candidate entities")
        (
            candidate_encoding,
            title2id,
            id2title,
            id2text,
            wikipedia_id2local_id,
            faiss_indexer,
        ) = _load_candidates(
            args.entity_catalogue,
            args.entity_encoding,
            faiss_index=args.faiss_index,
            index_path=args.index_path,
            logger=logger,
        )

        self.biencoder = biencoder
        self.biencoder_params = biencoder_params
        self.crossencoder = crossencoder
        self.crossencoder_params = crossencoder_params
        self.candidate_encoding = candidate_encoding
        self.title2id = title2id
        self.id2title = id2title
        self.id2text = id2text
        self.wikipedia_id2local_id = wikipedia_id2local_id
        self.faiss_indexer = faiss_indexer
        self.id2url = {
            v: "https://en.wikipedia.org/wiki?curid=%s" % k
            for k, v in wikipedia_id2local_id.items()
        }

    def link_text(self, text):
        # Identify mentions
        samples = _annotate(self.ner_model, [text])

        _print_colorful_text(text, samples)

        # don't look at labels
        keep_all = True

        # prepare the data for biencoder
        logger.info("preparing data for biencoder")
        dataloader = _process_biencoder_dataloader(
            samples, self.biencoder.tokenizer, self.biencoder_params
        )

        # run biencoder
        logger.info("run biencoder")
        top_k = args.top_k
        labels, nns, scores = _run_biencoder(
            self.biencoder,
            dataloader,
            self.candidate_encoding,
            top_k,
            self.faiss_indexer,
        )

        # print biencoder prediction
        idx = 0
        linked_entities = []
        for entity_list, sample, score_list in zip(nns, samples, scores):
            e_id = entity_list[0]
            e_title = self.id2title[e_id]
            e_text = self.id2text[e_id]
            e_url = self.id2url[e_id]
            e_score = float(score_list[0])
            linked_entities.append(
                {
                    "idx": idx,
                    "sample": sample,
                    "entity_id": e_id.item(),
                    "entity_title": e_title,
                    "entity_text": e_text,
                    "entity_score": e_score,
                    "url": e_url,
                    "crossencoder": False,
                }
            )
            idx += 1

        if args.fast:
            # use only biencoder
            return {"samples": samples, "linked_entities": linked_entities}

        # prepare crossencoder data
        context_input, candidate_input, label_input = prepare_crossencoder_data(
            self.crossencoder.tokenizer,
            samples,
            labels,
            nns,
            self.id2title,
            self.id2text,
            keep_all,
        )

        context_input = modify(
            context_input, candidate_input, self.crossencoder_params["max_seq_length"]
        )

        dataloader = _process_crossencoder_dataloader(
            context_input, label_input, self.crossencoder_params
        )

        # run crossencoder and get accuracy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accuracy, index_array, unsorted_scores = _run_crossencoder(
            self.crossencoder,
            dataloader,
            logger,
            context_len=self.biencoder_params["max_context_length"],
            device=device,
        )

        # print crossencoder prediction
        idx = 0
        linked_entities = []
        for entity_list, index_list, sample, unsorted_score_list in zip(
            nns, index_array, samples, unsorted_scores
        ):
            e_id = entity_list[index_list[-1]]
            e_title = self.id2title[e_id]
            e_text = self.id2text[e_id]
            e_url = self.id2url[e_id]
            e_score = float(unsorted_score_list[-1])
            _print_colorful_prediction(
                idx, sample, e_id, e_title, e_text, e_url, args.show_url
            )
            linked_entities.append(
                {
                    "idx": idx,
                    "sample": sample,
                    "entity_id": e_id.item(),
                    "entity_title": e_title,
                    "entity_text": e_text,
                    "entity_score": e_score,
                    "url": e_url,
                    "crossencoder": True,
                }
            )
            idx += 1
        return {"samples": samples, "linked_entities": linked_entities}

    def get_coref_resolved_text(self, texts):
        docs = [self.nlp(text) for text in texts]
        lengths = [len(list(doc.sents)) for doc in docs]
        text = ' '.join(texts)
        doc = self.nlp(text)
        entities = list([ent for ent in doc.ents if ent.label_ == 'PERSON'])

        for ent in entities:
            if ent._.coref_cluster:
                continue

            mentions = [ent]

            for _ent in entities:
                if ent == _ent:
                    continue

                cond = ent.text.startswith(_ent.text) or ent.text.endswith(_ent.text)
                cond = cond or _ent.text.startswith(ent.text) or _ent.text.endswith(ent.text)
                if cond:
                    mentions.append(_ent)
                    ent._.coref_cluster = ent._.coref_cluster or _ent._.coref_cluster
                    if ent._.coref_cluster:
                        # Check if ent span is already covered by existing mention
                        flag = False
                        for mention in ent._.coref_cluster.mentions:
                            if mention.start <= ent.start and mention.end >= ent.end:
                                flag = True
                                break
                        if not flag:
                            ent._.coref_cluster.mentions.append(ent)
                        break

            if ent._.coref_cluster is None and len(mentions) > 1:
                mentions = sorted(mentions, key=lambda x: x.start)
                cluster = neuralcoref.neuralcoref.Cluster(1, mentions[0], mentions)
                for mention in mentions:
                    mention._.coref_cluster = cluster
                
        clusters = list(set([ent._.coref_cluster for ent in entities if ent._.coref_cluster]))
        for cluster in clusters:
            cluster.mentions = sorted(cluster.mentions, key=lambda ent: len(ent.text), reverse=True)
            cluster.main = cluster.mentions[0]
        doc._.coref_clusters = clusters
        doc._.coref_resolved = neuralcoref.neuralcoref.get_resolved(doc, doc._.coref_clusters)

        text = doc._.coref_resolved
        doc = self.nlp(text)
        sents = list(doc.sents)
        sents = [sent.text for sent in sents]
        lengths.insert(0, 0)
        lengths = np.cumsum(lengths)
        sent_spans = [sents[lengths[i]:lengths[i + 1]] for i in range(len(lengths) - 1)]
        texts = [' '.join(sent_span) for sent_span in sent_spans]
        return texts

    def run(self):
        self.logger.info("interactive mode")
        while True:
            # Interactive
            text = input("insert text:")
            output = self.link_text(text)
            samples = output["samples"]
            linked_entities = output["linked_entities"]
            _print_colorful_text(text, samples)
            for e in linked_entities:
                _print_colorful_prediction(
                    e["idx"],
                    e["sample"],
                    e["entity_id"],
                    e["entity_title"],
                    e["entity_text"],
                    e["url"],
                    self.args.show_url,
                )
            print()

    def create_app(self):
        app = FastAPI()

        @app.post("/api/entity-link/single")
        async def entity_link(text: str):
            return self.link_text(text)

        @app.post("/api/resolve-coref")
        async def resolve_coref(texts: List[str]):
            return self.get_coref_resolved_text(texts)

        return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ner
    parser.add_argument(
        "--ner_config", 
        dest="ner_config", 
        type=str, 
        default="models/ner.json",
        help="Path to the NER configuration."
    )

    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
    )

    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=10,
        help="Number of candidates retrieved by biencoder.",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="only biencoder mode"
    )

    parser.add_argument(
        "--show_url",
        dest="show_url",
        action="store_true",
        help="whether to show entity url in interactive mode",
    )

    parser.add_argument(
        "--faiss_index", type=str, default=None, help="whether to use faiss index",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="path to load indexer",
    )

    parser.add_argument("--mode", type=str, default="interactive")

    args = parser.parse_args()

    logger = utils.get_logger(args.output_path)

    linker = EntityLinker(args, logger)
    if args.mode == "interactive":
        linker.run()
    elif args.mode == "api":
        app = linker.create_app()
        uvicorn.run(app, host="0.0.0.0", port=3030)
    else:
        raise ValueError("Invalid mode")
