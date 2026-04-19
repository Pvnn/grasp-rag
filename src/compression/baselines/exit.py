"""
Compressor adapted from the EXIT repository.

Repository:
https://github.com/ThisIsHwang/EXIT

Paper:
EXIT: Extractive Context Compression for Retrieval-Augmented Generation
Hwang et al., 2024
"""

import torch
import spacy
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from functools import lru_cache

from src.eval.interfaces import BaseCompressor, SearchResult


class EXITCompressor(BaseCompressor):
    """EXIT: Context-aware extractive compression."""

    def __init__(
        self,
        token: str = None,
        base_model: str = "google/gemma-2b-it",
        checkpoint: str = None,
        device: str = None,
        cache_dir: str = "./cache",
        batch_size: int = 2,
        threshold: float = 0.5
    ):
        self.batch_size = batch_size
        self.threshold = threshold

        # Match official: auto device map, not manual device string
        self.device = device or "auto"

        print(f"Initializing EXITCompressor with {base_model}...")

        # Tokenizer — identical to official
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 4-bit quantization via BitsAndBytesConfig — avoids deprecation warning
        # from passing load_in_4bit directly to from_pretrained
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=self.device,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            token=token
        )

        if checkpoint:
            self.peft_config = PeftConfig.from_pretrained(checkpoint, token=token)
            self.model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint,
                token=token
            )
        else:
            self.model = self.base_model

        self.model.eval()

        # Cache actual device after model is placed — same as official
        self.device = next(self.model.parameters()).device

        # Pre-cache token IDs — same as official
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        torch.cuda.empty_cache()

        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
        )
        self.nlp.enable_pipe("senter")

        print(f"✓ EXITCompressor ready on {self.device}")

    @lru_cache(maxsize=1024)
    def _generate_prompt(self, query: str, context: str, sentence: str) -> str:
        """Cached prompt generation — with Safe Context Truncation."""
        max_context_chars = 12000
        safe_context = context[:max_context_chars] + ("..." if len(context) > max_context_chars else "")

        return (
            f'<start_of_turn>user\n'
            f'Query:\n{query}\n'
            f'Full context:\n{safe_context}\n'
            f'Sentence:\n{sentence}\n'
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f'<start_of_turn>model\n'
        )

    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Predict relevance for a batch of (query, context, sentence) triples.
        Signature and return type identical to official impl.
        Returns (predictions list of 'Yes'/'No', probs tensor of shape [B, 2]).
        """
        prompts = [
            self._generate_prompt(q, c, s)
            for q, c, s in zip(queries, contexts, sentences)
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            return_attention_mask=True
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad():
            # Modern autocast syntax — avoids deprecation warning
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = self.model(**inputs)

                next_token_logits = outputs.logits[:, -1, :]
                relevant_logits = torch.stack([
                    next_token_logits[:, self.yes_token_id],
                    next_token_logits[:, self.no_token_id]
                ], dim=1)

                probs = torch.softmax(relevant_logits, dim=1)
                predictions = [
                    "Yes" if p else "No"
                    for p in probs.argmax(dim=1).cpu().numpy()
                ]

        return predictions, probs

    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Compress documents using context-aware extraction, evaluating 
        each sentence against its specific parent document to prevent truncation.
        """
        all_sentences = []
        all_contexts = []
        
        # Process document by document to map sentences to their parent text
        for doc in documents:
            doc_context = f"{doc.title}\n{doc.text}" if doc.title else doc.text
            
            # Extract sentences just for this document
            sents = [
                sent.text.strip()
                for sent in self.nlp(doc_context).sents
                if sent.text.strip()
            ]
            
            all_sentences.extend(sents)
            # Map each sentence to its specific parent document context
            all_contexts.extend([doc_context] * len(sents))

        if not all_sentences:
            return [SearchResult(evi_id=0, docid=0, title="", text="", score=1.0)]
            
        selected_sentences = []

        # Batch process using the perfectly mapped parent contexts
        for i in range(0, len(all_sentences), self.batch_size):
            batch_sents = all_sentences[i : i + self.batch_size]
            batch_queries = [query] * len(batch_sents)
            batch_contexts = all_contexts[i : i + self.batch_size]

            _, probs = self._predict_batch(batch_queries, batch_contexts, batch_sents)

            for sent, yes_prob in zip(batch_sents, probs[:, 0].tolist()):
                if yes_prob >= self.threshold:
                    selected_sentences.append(sent)

        compressed_text = " ".join(selected_sentences)

        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=compressed_text,
            score=1.0
        )]