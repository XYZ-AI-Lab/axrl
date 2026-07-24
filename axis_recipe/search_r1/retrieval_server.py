"""Copied and adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py."""

from __future__ import annotations

import argparse
import json
import warnings
from abc import ABC, abstractmethod
from typing import Any, cast

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Type aliases for retrieval results
Document = dict[str, Any]
SingleSearchResults = list[Document]
SearchResultsWithScores = tuple[SingleSearchResults, list[float]]
BatchSearchResults = list[SingleSearchResults]
BatchSearchResultsWithScores = tuple[BatchSearchResults, list[list[float]]]
QueryInput = str | list[str]


def load_corpus(corpus_path: str) -> Any:
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def read_jsonl(file_path: str) -> list[Any]:
    data: list[Any] = []
    with open(file_path) as f:  # noqa: PTH123
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus: datasets.Dataset, doc_idxs: list[int]) -> SingleSearchResults:
    results: SingleSearchResults = [cast("Document", corpus[int(idx)]) for idx in doc_idxs]
    return results


def load_model(model_path: str, *, use_fp16: bool = False) -> tuple[Any, Any]:
    # model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(
    pooler_output: torch.Tensor,
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    pooling_method: str = "mean",
) -> torch.Tensor:
    if pooling_method == "mean":
        if attention_mask is None:
            raise ValueError("attention_mask is required for mean pooling")
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name: str, model_path: str, pooling_method: str, max_length: int, *, use_fp16: bool) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: str | list[str], *, is_query: bool = True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(inputs["input_ids"].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class BaseRetriever(ABC):
    def __init__(self, config: Any) -> None:
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    @abstractmethod
    def _search(self, query: str, num: int | None, *, return_score: bool = False) -> SearchResultsWithScores | SingleSearchResults:
        raise NotImplementedError

    @abstractmethod
    def _batch_search(
        self, query_list: QueryInput, num: int | None, *, return_score: bool = False
    ) -> BatchSearchResultsWithScores | BatchSearchResults:
        raise NotImplementedError

    def search(self, query: str, num: int | None = None, *, return_score: bool = False) -> SearchResultsWithScores | SingleSearchResults:
        return self._search(query, num, return_score=return_score)

    def batch_search(
        self, query_list: QueryInput, num: int | None = None, *, return_score: bool = False
    ) -> BatchSearchResultsWithScores | BatchSearchResults:
        return self._batch_search(query_list, num, return_score=return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self) -> bool:
        return self.searcher.doc(0).raw() is not None  # type: ignore

    def _search(self, query: str, num: int | None = None, *, return_score: bool = False) -> SearchResultsWithScores | SingleSearchResults:
        if num is None:
            num = self.topk
        assert num is not None
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]  # type: ignore
            results = [
                {"title": content.split("\n")[0].strip('"'), "text": "\n".join(content.split("\n")[1:]), "contents": content}
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        return results

    def _batch_search(
        self, query_list: QueryInput, num: int | None = None, *, return_score: bool = False
    ) -> BatchSearchResultsWithScores | BatchSearchResults:
        if isinstance(query_list, str):
            query_list = [query_list]

        results: BatchSearchResults = []
        scores: list[list[float]] = []
        for query in query_list:
            item_result, item_score = self._search(query, num, return_score=True)
            results.append(cast("SingleSearchResults", item_result))
            scores.append(cast("list[float]", item_score))
        if return_score:
            return results, scores
        return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            if hasattr(faiss, "GpuMultipleClonerOptions"):
                co = faiss.GpuMultipleClonerOptions()  # type: ignore
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            else:
                warnings.warn(
                    "faiss_gpu=True but faiss-gpu is not installed (only faiss-cpu found). "
                    "Falling back to CPU index. To enable GPU, install faiss-gpu: "
                    "pip install faiss-gpu",
                    stacklevel=2,
                )

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int | None = None, *, return_score: bool = False) -> SearchResultsWithScores | SingleSearchResults:
        if num is None:
            num = self.topk
        assert num is not None
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        return results

    def _batch_search(
        self, query_list: QueryInput, num: int | None = None, *, return_score: bool = False
    ) -> BatchSearchResultsWithScores | BatchSearchResults:
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        assert num is not None

        results: BatchSearchResults = []
        scores: list[list[float]] = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = [idx for batch in batch_idxs for idx in batch]
            flat_results = load_docs(self.corpus, flat_idxs)
            split_batch_results: BatchSearchResults = [flat_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(split_batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, flat_results, split_batch_results
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        return results


def get_retriever(config: Config) -> BaseRetriever:
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """Minimal config class (simulating your argparse).

    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_batch_size: int = 128,
        *,
        faiss_gpu: bool = False,
        retrieval_use_fp16: bool = False,
    ) -> None:
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


app = FastAPI()

# Globals are initialized under __main__ for FastAPI use
config: Config | None = None
retriever: BaseRetriever | None = None


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest) -> dict[str, Any]:
    """Retrieve documents for the provided queries.

    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if retriever is None or config is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    topk = request.topk or config.retrieval_topk

    if request.return_scores:
        results, scores = cast("BatchSearchResultsWithScores", retriever.batch_search(query_list=request.queries, num=topk, return_score=True))
    else:
        results = cast("BatchSearchResults", retriever.batch_search(query_list=request.queries, num=topk, return_score=False))
        scores = None

    resp: list[Any] = []
    for i, single_result in enumerate(results):
        if request.return_scores and scores is not None:
            paired = [{"document": doc, "score": score} for doc, score in zip(single_result, scores[i], strict=True)]
            resp.append(paired)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")

    args = parser.parse_args()

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # 3) Launch the server.
    import os

    assert "AXRL_SEARCH_PORT" in os.environ, "Environment variable AXRL_SEARCH_PORT must be set."
    port = int(os.environ["AXRL_SEARCH_PORT"])
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", access_log=False)  # noqa: S104
