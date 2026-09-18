"""
FILCO-style Context Filter (Zero-shot CXMI approximation)

Paper: Learning to Filter Context for Retrieval-Augmented Generation
Wang et al., 2023 — https://arxiv.org/abs/2311.08377

Note: The original FILCO fine-tunes a FlanT5-xl checkpoint per dataset.
This implementation uses the zero-shot string-inclusion (strinc) filtering
strategy from the same paper, which does not require training and serves
as a fair approximation for cross-dataset evaluation.
"""

import re
import spacy
from typing import List

from src.eval.interfaces import BaseCompressor, SearchResult


class FILCoCompressor(BaseCompressor):
    """
    FILCO zero-shot variant using string inclusion filtering.
    Keeps sentences that contain at least one token from the query
    above a minimum overlap threshold — the 'strinc' strategy from the paper.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.15,
        min_keep: int = 2,
    ):
        self.overlap_threshold = overlap_threshold
        self.min_keep = min_keep

        print("Initializing FILCo (zero-shot strinc)...")
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        )
        self.nlp.enable_pipe("senter")
        print("✓ FILCo ready")

    def _query_tokens(self, query: str) -> set:
        """Lowercase non-stopword query tokens."""
        doc = self.nlp(query.lower())
        return {
            token.text for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        }

    def _overlap_score(self, sentence: str, query_tokens: set) -> float:
        """Fraction of query tokens present in the sentence."""
        if not query_tokens:
            return 0.0
        sent_lower = sentence.lower()
        hits = sum(1 for t in query_tokens if t in sent_lower)
        return hits / len(query_tokens)

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:
        # Build full context and split into sentences
        context = "\n".join(
            f"{doc.title}\n{doc.text}" if doc.title else doc.text
            for doc in documents
            if doc.text.strip()
        )

        if not context.strip():
            return []

        sentences = [
            s.text.strip()
            for s in self.nlp(context).sents
            if s.text.strip()
        ]

        query_tokens = self._query_tokens(query)

        # Score each sentence by query token overlap
        scored = [
            (sent, self._overlap_score(sent, query_tokens))
            for sent in sentences
        ]

        # Keep sentences above threshold, guarantee min_keep
        sorted_by_score = sorted(scored, key=lambda x: x[1], reverse=True)

        selected = []
        for i, (sent, score) in enumerate(sorted_by_score):
            if score >= self.overlap_threshold or i < self.min_keep:
                selected.append(sent)

        # Reconstruct in original document order
        selected_set = set(selected)
        ordered = [s for s, _ in scored if s in selected_set]

        compressed_text = " ".join(ordered)

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="filco",
                text=compressed_text,
                score=1.0,
            )
        ]