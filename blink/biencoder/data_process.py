# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import bz2
import logging
import torch
from tqdm import tqdm, trange
from glob import glob
from torch.utils.data import DataLoader, TensorDataset

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):
        # This try-except block is for the case that the mention tokens are exactly 30 in number
        # Read more about this issue here: https://github.com/facebookresearch/BLINK/issues/57
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)
        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )
        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx)
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)
    return data, tensor_data


import os
import io
import json
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset


class MentionDataset(IterableDataset):
    def __init__(self, 
        dataset_name, 
        preprocessed_json_data_parent_folder,
        tokenizer,
        max_context_length,
        max_cand_length,
        silent,
        mention_key="mention",
        context_key="context",
        label_key="label",
        title_key='label_title',
        ent_start_token=ENT_START_TAG,
        ent_end_token=ENT_END_TAG,
        title_token=ENT_TITLE_TAG,
        debug=False,
        logger=None,
    ) -> None:
        super().__init__()
        self.silent = silent
        self.debug = debug
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.mention_key = mention_key
        self.context_key = context_key
        self.ent_start_token = ent_start_token
        self.ent_end_token = ent_end_token
        self.label_key = label_key
        self.title_key = title_key
        self.max_cand_length = max_cand_length

        # file_name = "{}.jsonl".format(dataset_name)
        file_name = "{}.jsonl.bz2".format(dataset_name)
        self.txt_file_path = os.path.join(preprocessed_json_data_parent_folder, file_name)

        # with io.open(self.txt_file_path, mode="r", encoding="utf-8") as file:
        with bz2.open(self.txt_file_path, mode="rt", encoding="utf-8") as file:
            for idx, _ in enumerate(file):
                pass
            idx += 1
            # idx = idx // 3
            self.len = idx

    def __len__(self):
        return self.len

    def __iter__(self):
        use_world = True

        # Get total number of workers and worker id
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        num_workers = max(num_workers, 1)

        # with io.open(self.txt_file_path, mode="r", encoding="utf-8") as file:
        with bz2.open(self.txt_file_path, mode="rt", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                # if idx%3 != 0:
                #     continue
                if (idx+1)%num_workers != worker_id:
                    continue

                sample = json.loads(line.strip())
                try:
                    context_tokens = get_context_representation(
                        sample,
                        self.tokenizer,
                        self.max_context_length,
                        self.mention_key,
                        self.context_key,
                        self.ent_start_token,
                        self.ent_end_token,
                    )
                except AssertionError as e:
                    continue

                label = sample[self.label_key]
                title = sample.get(self.title_key, None)
                label_tokens = get_candidate_representation(
                    label, self.tokenizer, self.max_cand_length, title,
                )
                label_idx = int(sample["label_id"])

                record = {
                    "context": context_tokens,
                    "label": label_tokens,
                    "label_idx": [label_idx],
                }

                if "world" in sample:
                    src = sample["world"]
                    src = world_to_id[src]
                    record["src"] = [src]
                    use_world = True
                else:
                    use_world = False

                context_vec = torch.tensor(
                    record['context']['ids'], dtype=torch.long,
                )
                cand_vec = torch.tensor(
                    record['label']['ids'], dtype=torch.long,
                )
                if use_world:
                    src_vec = torch.tensor(
                        record['src'], dtype=torch.long,
                    )
                label_idx = torch.tensor(
                    record['label_idx'], dtype=torch.long,
                )

                if use_world:
                    yield (context_vec, cand_vec, src_vec, label_idx)
                else:
                    yield (context_vec, cand_vec, label_idx)
