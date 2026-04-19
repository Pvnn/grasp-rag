"""
RECOMP Abstractive Compressor

Adapted from the EXIT repository:
https://github.com/ThisIsHwang/EXIT

Paper:
EXIT: Extractive Context Compression for Retrieval-Augmented Generation
"""

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.eval.interfaces import BaseCompressor, SearchResult


class RECOMPAbstractiveCompressor(BaseCompressor):
    """
    RECOMP: Abstractive document compression using seq2seq models.
    """

    def __init__(
        self,
        model_name: str = "fangyuan/nq_abstractive_compressor",
        cache_dir: str = "./cache",
        max_length: int = 4096,
        device: str = None
    ):

        self.max_length = max_length

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else torch.device(device)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=torch.float16 
        ).to(self.device)

        self.model.eval()

    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:

        passage = "\n".join([doc.text for doc in documents])

        prompt = f"Question: {query}\nDocument: {passage}\nSummary: "

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=300,
                early_stopping=True,
                num_beams=5,
                length_penalty=2.0
            )

            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="recomp_abst",
                text=summary,
                score=1.0
            )
        ]