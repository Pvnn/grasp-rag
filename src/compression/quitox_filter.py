import torch
import numpy as np
from typing import List, Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration
from scipy.ndimage import gaussian_filter1d

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class QuitoxCoarseFilter:
    """
    Module 3: QUITO-X Coarse Filter (Min-Max Normalized Dynamic Thresholding)
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading QUITO-X model ({model_name}) on device: {self.device.upper()}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    def _softmax(self, x, axis=0):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def _get_sentence_scores(self, query: str, sentences: List[str]) -> Tuple[List[float], int]:
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        
        flat_ids = []
        boundaries = []
        
        for idx, sent in enumerate(sentences):
            prefix = " " if idx > 0 and not sent.startswith(" ") else ""
            ids = self.tokenizer.encode(prefix + sent, add_special_tokens=False)
            start_idx = len(flat_ids)
            end_idx = start_idx + len(ids)
            boundaries.append((start_idx, end_idx))
            flat_ids.extend(ids)

        max_chunk = 510 - len(query_ids)
        token_attentions = np.zeros(len(flat_ids))
        total_tokens_consumed = 0
        
        # 1. Group tokens into structural chunks
        all_chunk_ids = []
        for i in range(0, len(flat_ids), max_chunk):
            all_chunk_ids.append(flat_ids[i : i + max_chunk])
            
        if not all_chunk_ids:
            return [0.0] * len(sentences), 0

        # 2. Process Chunks in Parallel Batches
        batch_size = 16  # T5-small easily fits 16 chunks in VRAM
        
        for i in range(0, len(all_chunk_ids), batch_size):
            batch_chunks = all_chunk_ids[i : i + batch_size]
            
            # Prepare inputs with query + EOS
            batch_input_ids = [chunk + query_ids + [self.tokenizer.eos_token_id] for chunk in batch_chunks]
            
            # Dynamic Padding for the batch
            max_len = max(len(ids) for ids in batch_input_ids)
            padded_inputs = [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in batch_input_ids]
            attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_input_ids]

            # Move tensors to GPU
            input_tensor = torch.tensor(padded_inputs).to(self.device, non_blocking=True)
            mask_tensor = torch.tensor(attention_mask).to(self.device, non_blocking=True)
            decoder_input = torch.tensor([[self.model.config.decoder_start_token_id]] * len(batch_chunks)).to(self.device, non_blocking=True)
            
            total_tokens_consumed += sum(len(ids) for ids in batch_input_ids)

            # FIX 3: Hardware-accelerated Autocast
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outputs = self.model(
                    input_ids=input_tensor,
                    attention_mask=mask_tensor,
                    decoder_input_ids=decoder_input,
                    output_attentions=True
                )
        
            # Extract Cross-Attentions (Mean across all heads)
            # Shape: (Batch, Seq_Len)
            attentions = outputs.cross_attentions[-1][:, :, 0, :].mean(dim=1).cpu().numpy()
            
            # Map batch attentions back to the global token array
            for j, chunk_attn in enumerate(attentions):
                global_idx = (i + j) * max_chunk
                valid_len = len(batch_chunks[j])
                token_attentions[global_idx : global_idx + valid_len] = chunk_attn[:valid_len]

        # 3. Apply smoothing to the contiguous token array
        if len(token_attentions) > 0:
            smoothed_attentions = gaussian_filter1d(token_attentions, sigma=2.0)
        else:
            smoothed_attentions = token_attentions

        # 4. Extract Max token-score per sentence using our exact boundaries
        sent_scores = []
        for start, end in boundaries:
            segment = smoothed_attentions[start:end]
            score = np.max(segment) if len(segment) > 0 else 0.0
            sent_scores.append(float(score))

        return sent_scores, total_tokens_consumed

    def compress(self, query: str, sentences: List[str], compression_ratio: float = 0.6, min_keep: int = 2) -> dict:
        if not sentences:
            return {"filtered_sentences": [], "kept_indices": [], "total_tokens_consumed": 0, "quitox_details": []}

        raw_scores, total_tokens = self._get_sentence_scores(query, sentences)
        scored_sentences = [{"text": s, "score": score} for s, score in zip(sentences, raw_scores)]

        # Top-K ratio threshold — same as old impl
        sorted_scores = sorted([s['score'] for s in scored_sentences], reverse=True)
        num_keep = max(min_keep, int(len(sentences) * compression_ratio))  # ← min_keep folded in here
        threshold_idx = min(num_keep - 1, len(sorted_scores) - 1)
        keep_threshold = sorted_scores[threshold_idx] if sorted_scores else 0.0

        filtered_sentences = []
        indices_to_keep = []
        quitox_details = []

        for idx, s in enumerate(scored_sentences):
            is_retained = s['score'] >= keep_threshold and len(filtered_sentences) < num_keep
            if is_retained:
                filtered_sentences.append(s['text'])
                indices_to_keep.append(idx)

            quitox_details.append({
                "text": s['text'],
                "score": round(s['score'], 4),
                "retained": is_retained
            })

        return {
            "filtered_sentences": filtered_sentences,
            "kept_indices": indices_to_keep,        # ← needed by HybridCompressor for doc-map sync
            "total_tokens_consumed": total_tokens,
            "quitox_details": quitox_details
        }