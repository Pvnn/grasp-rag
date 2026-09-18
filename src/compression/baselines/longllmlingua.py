"""
LongLLMLingua Compressor Adapter
"""
import torch
import spacy

from typing import List

from llmlingua import PromptCompressor
from sentence_transformers import CrossEncoder

from src.eval.interfaces import BaseCompressor, SearchResult

class LongLLMLinguaCompressor(BaseCompressor):
    """
    LongLLMLingua: extends LLMLingua for long-context RAG
    by adding document-level reordering and a question-aware
    coarse filter before token-level compression.

    Paper: Jiang et al., ACL 2024
    https://arxiv.org/abs/2310.06839
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        compression_rate: float = 0.4,
    ):
        self.compression_rate = compression_rate

        print(f"Initializing LongLLMLingua with {model_name}...")

        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=False,
            device_map=device_map,
            model_config={"cache_dir": "./cache"},
        )

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:

        chunked_context = []

        for doc in documents:
            sentences = [
                s.strip() + "."
                for s in doc.text.split(".")
                if s.strip()
            ]

            chunked_context.extend(sentences)

        if not chunked_context:
            return []

        total_words = sum(
            len(c.split()) for c in chunked_context
        )

        target = max(
            1,
            int(total_words * self.compression_rate)
        )

        result = self.compressor.compress_prompt(
            context=chunked_context,
            question=query,
            target_token=target,
            rank_method="longllmlingua",
            dynamic_context_compression_ratio=0.3,
            condition_compare=True,
            context_budget="+100",
        )

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="longllmlingua",
                text=result["compressed_prompt"],
                score=1.0,
            )
        ]