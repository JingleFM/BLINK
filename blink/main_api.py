# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
import sys

from tqdm import tqdm
import logging
import torch
import numpy as np
from colorama import init
from termcolor import colored

import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.crossencoder.train_cross import modify, evaluate
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.indexer.faiss_indexer import FaissIndexer


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
        print("Failed to identify entity from text:")
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\n".format(e_id)
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
    entity_encoding, faiss_index=None, index_path=None, logger=None
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
        indexer = FaissIndexer(faiss_index, 1024)
        indexer.deserialize_from(index_path)

    return (
        candidate_encoding,
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
    device = biencoder.device
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        context_input = context_input.to(device)
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding.to(device)
                )
                scores, indicies = scores.topk(top_k)
                scores = scores.data.numpy()
                indicies = indicies.data.numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores


def load_models(args, logger=None):

    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    (
        candidate_encoding,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_encoding, 
        faiss_index=getattr(args, 'faiss_index', None), 
        index_path=getattr(args, 'index_path' , None),
        logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        candidate_encoding,
        faiss_indexer,
    )


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    candidate_encoding,
    faiss_indexer=None,
):
    stopping_condition = False
    while not stopping_condition:

        samples = None

        if args.interactive:
            logger.info("interactive mode")

            # biencoder_params["eval_batch_size"] = 1

            # Load NER model
            ner_model = NER.get_model()

            # Interactive
            text = input("insert text:")

            # Identify mentions
            samples = _annotate(ner_model, [text])

            _print_colorful_text(text, samples)

        # don't look at labels
        keep_all = (
            args.interactive
            or samples[0]["label"] == "unknown"
            or samples[0]["label_id"] < 0
        )

        # prepare the data for biencoder
        if logger:
            logger.info("preparing data for biencoder")
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        # run biencoder
        if logger:
            logger.info("run biencoder")
        top_k = args.top_k
        labels, nns, scores = _run_biencoder(
            biencoder, dataloader, candidate_encoding, top_k, faiss_indexer
        )

        if args.interactive:

            print("\nfast (biencoder) predictions:")

            _print_colorful_text(text, samples)

            # print biencoder prediction
            idx = 0
            for entity_list, sample in zip(nns, samples):
                e_id = entity_list[0]
                _print_colorful_prediction(
                    idx, sample, e_id
                )
                idx += 1
            print()

        else:

            biencoder_accuracy = -1
            recall_at = -1
            if not keep_all:
                # get recall values
                top_k = args.top_k
                x = []
                y = []
                for i in range(1, top_k):
                    temp_y = 0.0
                    for label, top in zip(labels, nns):
                        if label in top[:i]:
                            temp_y += 1
                    if len(labels) > 0:
                        temp_y /= len(labels)
                    x.append(i)
                    y.append(temp_y)
                # plt.plot(x, y)
                biencoder_accuracy = y[0]
                recall_at = y[-1]
                print("biencoder accuracy: %.4f" % biencoder_accuracy)
                print("biencoder recall@%d: %.4f" % (top_k, y[-1]))


            predictions = []
            for entity_list in nns:
                sample_prediction = []
                for e_id in entity_list:
                    sample_prediction.append(e_id)
                predictions.append(sample_prediction)

            # use only biencoder
            return (
                biencoder_accuracy,
                recall_at,
                -1,
                -1,
                len(samples),
                predictions,
                scores,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode."
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
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
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
        "--faiss_index", type=str, default=None, help="faiss index factory",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="path to load indexer",
    )

    args = parser.parse_args()

    logger = utils.get_logger(args.output_path)

    models = load_models(args, logger)
    run(args, logger, *models)
