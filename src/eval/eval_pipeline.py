import time
from typing import List, Dict
from tqdm import tqdm

from src.eval.interfaces import SearchResult
from src.eval.metrics import (
    answer_em_correctness, 
    answer_rouge_correctness, 
    DisambigF1Evaluator,
    answer_f1_correctness
)

class GenerativeEvaluator:
    def __init__(self, compressor, reader):
        self.compressor = compressor
        self.reader = reader
        self.disambig_eval = DisambigF1Evaluator()

    def evaluate(self, dataset: List[Dict], top_k: int = 10) -> Dict:
        metrics = {
            "em": 0.0,
            "token_f1": 0.0,
            "rougeL": 0.0,
            "disambig_f1": 0.0,
            "total_queries": len(dataset),
            "prompt_tokens": 0,
            "completion_tokens": 0
        }

        total_orig_chars = 0
        total_comp_chars = 0
        start_time = time.time()

        detailed_traces = []

        for item in tqdm(dataset, desc="Evaluating Pipeline"):
            query = item["question"]
            
            if "acceptable_answers" in item:
                gt_answers = item["acceptable_answers"]
            else:
                gt_answers = [item["answer"]]

            # 1. Load the PRE-RETRIEVED top-k documents from our offline script
            docs = []
            for i, context_item in enumerate(item["context"][:top_k]):
                if isinstance(context_item, str):
                    title = f"Doc {i+1}"
                    full_doc_text = context_item
                else:
                    title, sentences = context_item
                    full_doc_text = " ".join(sentences)
                    
                docs.append(SearchResult(evi_id=i, docid=i, title=title, text=full_doc_text))

            orig_text = "\n".join([d.text for d in docs])
            total_orig_chars += len(orig_text)

            # 2. Compress the docs
            result = self.compressor.compress(query, docs)
            compressed_docs = result.get("compressed_docs", [])
            comp_text = "\n".join([d.text for d in compressed_docs])
            total_comp_chars += len(comp_text)

            # 3. Generate Final Answer via LLM
            reader_res = self.reader.generate_answer(query, comp_text, strict_mode=True)
            
            if isinstance(reader_res, dict):
                pred_answer = reader_res.get("answer", "")
                usage = reader_res.get("usage", {})
            else:
                pred_answer = str(reader_res)
                usage = {}

            q_prompt_tokens = usage.get("prompt_tokens", 0)
            q_comp_tokens = usage.get("completion_tokens", 0)

            # 4. Calculate Query-Level Metrics (These inherently use the full list!)
            q_em = answer_em_correctness(pred_answer, gt_answers)
            q_token_f1 = answer_f1_correctness(pred_answer, gt_answers)
            q_rouge = answer_rouge_correctness(pred_answer, gt_answers, rouge_type="rougeL")
            q_f1 = self.disambig_eval.evaluate(pred_answer, gt_answers)

            metrics["em"] += q_em
            metrics["token_f1"] += q_token_f1
            metrics["rougeL"] += q_rouge
            metrics["disambig_f1"] += q_f1
            metrics["prompt_tokens"] += q_prompt_tokens
            metrics["completion_tokens"] += q_comp_tokens

            # 5. Build the Excel Row Data
            row_data = {
                "Query": query,
                "Gold Answer": str(gt_answers), 
                "Generated Answer": pred_answer,
                "EM Score": q_em,
                "Token F1": q_token_f1,
                "ROUGE-L Score": q_rouge,
                "Disambig-F1": q_f1,
                "Prompt Tokens": q_prompt_tokens,
                "Gen Tokens": q_comp_tokens,
                "Compression %": round((1 - (len(comp_text) / max(len(orig_text), 1))) * 100, 2),
                "Final Prompt Context": comp_text,
            }

            for doc_idx in range(top_k):
                row_data[f"Orig Doc {doc_idx+1}"] = docs[doc_idx].text if doc_idx < len(docs) else ""
                row_data[f"Comp Doc {doc_idx+1}"] = compressed_docs[doc_idx].text if doc_idx < len(compressed_docs) else ""

            detailed_traces.append(row_data)

        # Average out the metrics
        n = metrics["total_queries"]
        if n > 0:
            metrics["em"] = (metrics["em"] / n) * 100
            metrics["token_f1"] = (metrics["token_f1"] / n) * 100
            metrics["rougeL"] = (metrics["rougeL"] / n) * 100
            metrics["disambig_f1"] = (metrics["disambig_f1"] / n) * 100
            metrics["avg_prompt_tokens"] = metrics["prompt_tokens"] / n
            metrics["avg_completion_tokens"] = metrics["completion_tokens"] / n
        else:
            metrics["avg_prompt_tokens"] = 0.0
            metrics["avg_completion_tokens"] = 0.0

        if total_orig_chars > 0:
            metrics["compression_ratio_chars"] = (1 - (total_comp_chars / total_orig_chars)) * 100
        else:
            metrics["compression_ratio_chars"] = 0.0

        metrics["avg_latency_sec"] = (time.time() - start_time) / n if n > 0 else 0.0

        return {
            "aggregate": metrics,
            "details": detailed_traces
        }