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
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.indexer.faiss_indexer import FaissIndexer


HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]


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
        indexer = FaissIndexer(faiss_index, 1024)
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    id2entity = {}
    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            id2entity[entity["id"]] = {k:v for k,v in entity.items() if k in ["id", "title"]}
            local_idx += 1
    return (
        candidate_encoding,
        id2entity,
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
        id2entity,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue, 
        args.entity_encoding, 
        faiss_index=getattr(args, 'faiss_index', None), 
        index_path=getattr(args, 'index_path' , None),
        logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        candidate_encoding,
        id2entity,
        faiss_indexer,
    )


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    candidate_encoding,
    id2entity,
    faiss_indexer=None,
    test_data=None,
):

    if not test_data:
        msg = (
            "ERROR: you must start BLINK and "
            "pass in input test mentions (--test_mentions)"
        )
        raise ValueError(msg)

    samples = test_data

    # don't look at labels
    keep_all = (
        samples[0]["label"] == "unknown"
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
            entity = id2entity[e_id]
            sample_prediction.append(entity)
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
        "--faiss_index", type=str, default=None, help="whether to use faiss index",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="path to load indexer",
    )

    args = parser.parse_args()

    logger = utils.get_logger(args.output_path)

    models = load_models(args, logger)
    run(args, logger, *models)
