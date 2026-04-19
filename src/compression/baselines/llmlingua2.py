"""
LLMLingua-2 Compressor Adapter
"""

import torch
from typing import List
from llmlingua import PromptCompressor

from src.eval.interfaces import BaseCompressor, SearchResult


class LLMLingua2Compressor(BaseCompressor):
    """
    LLMLingua-2: Token-level context compression using extractive token classification.
    """

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        compression_rate: float = 0.2, 
    ):

        self.compression_rate = compression_rate

        print(f"Initializing LLMLingua-2 with {model_name}...")
        
        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map=device_map
        )

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:

        chunked_context = []
        
        # Chunk the documents into sentences so BERT's 512-token limit is never exceeded
        for doc in documents:
            sentences = [s.strip() + "." for s in doc.text.split(".") if s.strip()]
            chunked_context.extend(sentences)

        if not chunked_context:
            return []

        # Target token count based on the combined chunk length
        total_words = sum(len(c.split()) for c in chunked_context)
        target = max(1, int(total_words * self.compression_rate))

        result = self.compressor.compress_prompt(
            context=chunked_context,
            question=query,
            target_token=target,
        )

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="llmlingua-2",
                text=result["compressed_prompt"],
                score=1.0,
            )
        ]