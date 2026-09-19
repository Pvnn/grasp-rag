"""
JinaAI Reranker Compressor Adapter
"""

import torch
import spacy
import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_roberta

# Hotfix for transformers >= 4.45.0 deleting this function, causing Jina remote code to crash
if not hasattr(xlm_roberta, "create_position_ids_from_input_ids"):
    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx
    xlm_roberta.create_position_ids_from_input_ids = create_position_ids_from_input_ids

from typing import List

from llmlingua import PromptCompressor
from sentence_transformers import CrossEncoder

from src.eval.interfaces import BaseCompressor, SearchResult

class JinaRerankerCompressor(BaseCompressor):
    """
    JinaAI Reranker v2 used as a sentence-level compressor.
    Scores each sentence against the query via cross-encoder,
    keeps top-K sentences.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        compression_ratio: float = 0.5,
        min_keep: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.compression_ratio = compression_ratio
        self.min_keep = min_keep

        print(
            f"Initializing JinaAI Reranker Compressor with {model_name}..."
        )

        self.model = CrossEncoder(
            model_name,
            num_labels=1,
            trust_remote_code=True,
            device=device,
            cache_folder="./cache",
        )

        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=[
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        )

        self.nlp.enable_pipe("senter")

        print("✓ JinaAI Reranker Compressor ready")

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:

        context = "\n".join(
            f"{doc.title}\n{doc.text}"
            if doc.title
            else doc.text
            for doc in documents
            if doc.text.strip()
        )

        sentences = [
            s.text.strip()
            for s in self.nlp(context).sents
            if s.text.strip()
        ]

        if not sentences:
            return []

        pairs = [
            [query, s]
            for s in sentences
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        num_keep = max(
            self.min_keep,
            int(len(sentences) * self.compression_ratio),
        )

        scored = sorted(
            zip(scores, sentences),
            reverse=True,
        )

        top_sentences = {
            s for _, s in scored[:num_keep]
        }

        ordered = [
            s for s in sentences
            if s in top_sentences
        ]

        compressed_text = " ".join(ordered)

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="jina_reranker",
                text=compressed_text,
                score=1.0,
            )
        ]