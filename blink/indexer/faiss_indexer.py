# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
FAISS-based index components. Original from 
https://github.com/facebookresearch/DPR/blob/master/dpr/indexer/faiss_indexers.py
"""

import os
import logging
import pickle

import faiss
import numpy as np

from tqdm import tqdm

logger = logging.getLogger()


class FaissIndexer(object):
    def __init__(self, index_factory, vector_sz: int = 1, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index = faiss.index_factory(vector_sz, index_factory)

    def index_data(self, data: np.array):
        if not self.index.is_trained:
            logger.info("Index is not trained, training it.")
            self.build_index(data)
        
        n = len(data)
        logger.info("Indexing data, this may take a while.")
        cnt = 0
        for i in tqdm(range(0, n, self.buffer_size)):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + self.buffer_size]]
            vectors = np.concatenate(vectors, axis=0)
            self.index.add(vectors)
            cnt += self.buffer_size

        logger.info("Total data indexed %d", n)

    def build_index(self, data: np.array):
        logger.info("Building index, this may take a while.")
        self.index.train(data)

    def search_knn(self, query_vectors, top_k):
        scores, indexes = self.index.search(query_vectors, top_k)
        return scores, indexes

    def serialize(self, index_file: str):
        logger.info("Serializing index to %s", index_file)

        # We probe 1/10the of the Voronoi cells
        try:
            ivf = faiss.extract_index_ivf(self.index)
            ivf.nprobe = ivf.nlist // 10
            ivf.nprobe = max(1, ivf.nprobe)
        except:
            pass

        faiss.write_index(self.index, index_file)

    def deserialize_from(self, index_file: str):
        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )
