"""
Selective Context Compressor

Paper: Compressing Context to Enhance Inference Efficiency of Large Language Models
Li et al., EMNLP 2023
https://github.com/liyucheng09/Selective_Context

pip install selective-context
"""

from typing import List
import os
os.environ["HF_HOME"] = "./cache"
from selective_context import SelectiveContext

from src.eval.interfaces import BaseCompressor, SearchResult


class SelectiveContextCompressor(BaseCompressor):
    """
    Selective Context: perplexity-based extractive compression.
    Uses GPT-2 self-information to identify and remove low-information tokens.
    No query awareness — purely content-driven filtering.
    """

    def __init__(
        self,
        model_type: str = "gpt2",
        reduce_ratio: float = 0.35,
        lang: str = "en",
    ):
        self.reduce_ratio = reduce_ratio
        print(f"Initializing SelectiveContext with {model_type}, reduce_ratio={reduce_ratio}...")
        self.sc = SelectiveContext(model_type=model_type, lang=lang)
        print("✓ SelectiveContext ready")

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:
        # Concatenate all documents into one passage
        full_text = "\n".join(
            f"{doc.title}\n{doc.text}" if doc.title else doc.text
            for doc in documents
            if doc.text.strip()
        )

        if not full_text.strip():
            return []

        # SelectiveContext returns (compressed_text, reduced_content)
        # query is not used — this is the key difference from query-aware methods
        compressed_text, _ = self.sc(full_text, reduce_ratio=self.reduce_ratio)

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="selective_context",
                text=compressed_text,
                score=1.0,
            )
        ]