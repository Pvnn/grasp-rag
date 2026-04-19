from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import spacy
from functools import lru_cache
from typing import List, Tuple

class ExitBaselineCompressor:
    def __init__(
        self,
        token,
        model_name="doubleyyh/exit-gemma-2b",
        threshold=0.4,
        batch_size=2
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        print(f"Initializing ExitBaselineCompressor...")

        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available, using CPU")
        else:
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map={"": 0},
            token=token
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        # Safety buffer for spacy on large docs
        self.nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        self.nlp.enable_pipe("senter")
        self.nlp.max_length = 2000000  

    def decompose_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

    @lru_cache(maxsize=1024)
    def _generate_prompt(self, query: str, context: str, sentence: str) -> str:
        # Fallback safety truncation, though rarely hit now since we pass individual parent docs
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
        self, queries: List[str], contexts: List[str], sentences: List[str]
    ) -> Tuple[List[float], int]:
        """Now accepts a list of contexts so each sentence maps to its parent doc!"""
        prompts = [
            self._generate_prompt(q, c, s)
            for q, c, s in zip(queries, contexts, sentences)
        ]

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=4096, return_attention_mask=True
        )
        total_tokens = inputs["input_ids"].numel()
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outputs = self.model(**inputs)
            last_token_logits = outputs.logits[:, -1, :]
            relevant_logits = torch.stack([
                last_token_logits[:, self.yes_token_id],
                last_token_logits[:, self.no_token_id]
            ], dim=1)

            probs = torch.softmax(relevant_logits, dim=1)
            yes_probs = probs[:, 0].tolist()

        return yes_probs, total_tokens

    def compress(self, query: str, documents: list) -> list:
        """Updated baseline to evaluate using parent contexts."""
        all_sentences = []
        all_contexts = []
        
        for doc in documents:
            text = f"{doc.title}\n{doc.text}" if getattr(doc, 'title', None) else doc.text
            sents = self.decompose_sentences(text)
            all_sentences.extend(sents)
            all_contexts.extend([text] * len(sents)) # Map sentence to its specific parent text
            
        if not all_sentences:
            from src.eval.interfaces import SearchResult
            return [SearchResult(evi_id=0, docid=0, title="", text="", score=1.0)]
            
        selected_sentences = []
        for i in range(0, len(all_sentences), self.batch_size):
            batch_sents = all_sentences[i : i + self.batch_size]
            batch_queries = [query] * len(batch_sents)
            batch_contexts = all_contexts[i : i + self.batch_size]
            
            yes_probs, _ = self._predict_batch(batch_queries, batch_contexts, batch_sents)
            for sent, prob in zip(batch_sents, yes_probs):
                if prob >= self.threshold:
                    selected_sentences.append(sent)
        
        from src.eval.interfaces import SearchResult
        return [SearchResult(evi_id=0, docid=0, title="", text=" ".join(selected_sentences), score=1.0)]