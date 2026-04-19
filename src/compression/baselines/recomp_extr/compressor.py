"""
RECOMP Extractive Compressor

Adapted from EXIT repository:
https://github.com/ThisIsHwang/EXIT

Paper:
EXIT: Extractive Context Compression for Retrieval-Augmented Generation
"""

import torch
from typing import List
from transformers import AutoTokenizer

from src.eval.interfaces import BaseCompressor, SearchResult
from .contriever import Contriever


class RecompExtractiveCompressor(BaseCompressor):
    """
    RECOMP Extractive: selects most relevant passages using dense retrieval.
    """

    def __init__(
        self,
        model_name: str = "fangyuan/nq_extractive_compressor",
        batch_size: int = 32,
        cache_dir: str = "./cache",
        device: str = None,
        top_k_docs: int = 2 
    ):

        self.batch_size = batch_size
        self.top_k_docs = top_k_docs

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else torch.device(device)
        )

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        self.model = Contriever.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16
        ).to(self.device)

        self.model.eval()

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                embeddings = self.model(**inputs)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:

        if not documents:
            return []

        texts = [query] + [doc.text for doc in documents]
        embeddings = self.encode_texts(texts)

        query_embedding = embeddings[0].unsqueeze(0)  # Shape: (1, D)
        doc_embeddings = embeddings[1:]               # Shape: (N, D)

        similarity = (query_embedding @ doc_embeddings.T).squeeze(0)
        scores = similarity.cpu()

        k = min(self.top_k_docs, len(documents))
        
        top_indices = torch.topk(scores, k).indices.tolist()

        if not isinstance(top_indices, list):
            top_indices = [top_indices]

        compressed_docs = []
        for idx in top_indices:
            doc = documents[idx]
            compressed_docs.append(
                SearchResult(
                    evi_id=doc.evi_id,
                    docid=doc.docid,
                    title=doc.title,
                    text=doc.text,
                    score=float(scores[idx])
                )
            )

        return sorted(
            compressed_docs,
            key=lambda x: x.score,
            reverse=True
        )