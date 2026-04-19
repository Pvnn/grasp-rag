"""
CompAct Compressor Adapter

Adapted from:
EXIT repository
https://github.com/ThisIsHwang/EXIT

Paper:
EXIT: Extractive Context Compression for Retrieval-Augmented Generation
"""

import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.eval.interfaces import BaseCompressor, SearchResult


class CompactCompressor(BaseCompressor):
    """
    CompAct: Context compression using iterative summarization.
    """

    def __init__(
        self,
        token: str = None,
        model_dir: str = "cwyoon99/CompAct-7b",
        device: str = "cuda",
        cache_dir: str = "./cache",
        batch_size: int = 5,
    ):
        self.device = device
        self.batch_size = batch_size

        print(f"Initializing CompAct Compressor with {model_dir} in 4-bit...")

        # FIX: Resolves load_in_4bit deprecation warning
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=cache_dir,
            token=token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
            token=token
        )

    def _create_prompt(
        self,
        query: str,
        documents: str,
        prev_summary: str = "",
        prev_eval: str = "",
        iteration: int = 0,
    ) -> str:
        
        if iteration == 0:
            instruction = (
                "1. Generate a summary of source documents to answer the question. "
                "Ensure the summary is under 200 words and does not include any pronouns. "
                "DO NOT make assumptions or attempt to answer the question; "
                "your job is to summarize only.\n\n"
                "2. Evaluate the summary based solely on the information of it, "
                "without any additional background context: if it lacks sufficient "
                "details to answer the question, print '[INCOMPLETE]'. If it provides "
                "all necessary details, print '[COMPLETE]'. You should provide the "
                "reason of evalution."
            )
            prompt = f"{instruction}\n\nQuestion: {query}\n\nSource documents: {documents}\n\nSummary:"
        else:
            instruction = (
                "1. Generate a summary of the previous summary and the source documents "
                "to answer the question based on the evaluation of the previous summary. "
                "The evaluation indicates the missing information needed to answer the "
                "question. Ensure the summary is under 200 words and does not include "
                "any pronouns. DO NOT make assumptions or attempt to answer the question; "
                "your job is to summarize only.\n\n"
                "2. Evaluate the summary based solely on the information of it, without "
                "any additional background context: if it lacks sufficient details to "
                "answer the question, print '[INCOMPLETE]'. If it provides all necessary "
                "details, print '[COMPLETE]'. You should provide the reason of evalution."
            )
            prompt = (
                f"{instruction}\n\nQuestion: {query}\n\n"
                f"Previous summary: {prev_summary}\n\n"
                f"Evaluation of previous summary: {prev_eval}\n\n"
                f"Source documents: {documents}\n\nSummary:"
            )

        messages = [{"role": "user", "content": prompt}]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _parse_output(self, text: str) -> Dict[str, str]:
        summary_pattern = r"(Summary:)(.*?)(?=Evaluation:|$)"
        evaluation_pattern = r"(Evaluation:)(.*?)(?=Summary:|$)"

        summary_match = re.search(summary_pattern, text, re.DOTALL | re.IGNORECASE)
        eval_match = re.search(evaluation_pattern, text, re.DOTALL | re.IGNORECASE)

        summary = summary_match.group(2).strip() if summary_match else text.strip()
        evaluation = eval_match.group(2).strip() if eval_match else ""

        return {
            "summary": summary.replace("\n\n", ""),
            "eval": evaluation.replace("\n\n", ""),
        }

    def compress(
        self,
        query: str,
        documents: List[SearchResult],
    ) -> List[SearchResult]:

        prev_summaries = []
        prev_evals = []

        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i : i + self.batch_size]
            batch_text = "\n".join(
                f"{doc.title}\n{doc.text}" for doc in batch_docs
            )

            prev_summary = prev_summaries[-1] if prev_summaries else ""
            prev_eval = prev_evals[-1].replace('[INCOMPLETE]', '').strip() if prev_evals else ""

            prompt_text = self._create_prompt(
                query=query, 
                documents=batch_text, 
                prev_summary=prev_summary, 
                prev_eval=prev_eval,
                iteration=i // self.batch_size
            )

            with torch.no_grad():
                # FIX: Add truncation with max_length=1500 to leave room for the 500 generation tokens.
                # This guarantees it will never exceed the 2048 strict limit.
                inputs = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1500
                ).to(self.device)

                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=500,
                    do_sample=False, # FIX: Replaces deprecated temperature=0
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            output_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.size(1):],
                skip_special_tokens=True,
            ).strip()

            parsed = self._parse_output(output_text)
            prev_summaries.append(parsed["summary"])
            prev_evals.append(parsed["eval"])

            if "[COMPLETE]" in parsed["eval"].upper():
                break

        compressed_text = prev_summaries[-1] if prev_summaries else ""

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="compact",
                text=compressed_text,
                score=1.0,
            )
        ]