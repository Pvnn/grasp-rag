"""
Refiner Compressor

Adapted from the EXIT repository:
https://github.com/ThisIsHwang/EXIT
"""

import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.eval.interfaces import BaseCompressor, SearchResult

class RefinerCompressor(BaseCompressor):

    def __init__(
        self,
        token, 
        base_model="meta-llama/Llama-2-7b-chat-hf",
        adapter="al1231/Refiner-7B",
        device="cuda",
        max_tokens=3500,
        cache_dir="./cache"
    ):

        self.device = device
        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            token=token  
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=token 
        )

        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter,
            is_trainable=False,
            token=token  
        )

        self.model.eval()

        self.template = (
            "[INST]<<SYS>>"
            "[MONITOR]{context}"
            "<</SYS>>{question}[/INST] "
        )

    def _truncate_context(self, context: str) -> str:
        tokens = self.tokenizer.encode(
            context,
            add_special_tokens=False
        )

        if len(tokens) <= self.max_tokens:
            return context

        truncated = tokens[:self.max_tokens]

        return self.tokenizer.decode(
            truncated,
            skip_special_tokens=True
        )

    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:

        context = "\n".join(
            f"## {doc.title}\n{doc.text}"
            for doc in documents
            if doc.text.strip()
        )

        context = self._truncate_context(context)

        prompt = self.template.format(
            question=query,
            context=context
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():            
            outputs = self.model.generate(
                **inputs,
                top_p=1,
                temperature=None,
                do_sample=False,
                max_new_tokens=512
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        compressed_text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="refiner",
                text=compressed_text,
                score=1.0
            )
        ]