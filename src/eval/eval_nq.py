import sys
from pathlib import Path
import gc
import torch
import os
import pandas as pd 
import json
import re
import unicodedata
import argparse
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tabulate import tabulate

from src.eval.eval_pipeline import GenerativeEvaluator
from src.generation.reader import RAGReader

from src.eval.adapters import (
    NoOpCompressor, 
    ExitAdapter, 
    HybridAdapter,
    RefinerAdapter, 
    RecompAdapter, 
    LLMLingua2Adapter,
    CompActAdapter, 
    RecompExtractiveAdapter
)

# --- Import Compressors ---
from src.compression.hybrid_compressor import HybridCompressor
from src.compression.baselines import (
    EXITCompressor,
    RefinerCompressor,
    RECOMPAbstractiveCompressor,
    LLMLingua2Compressor,
    CompactCompressor,
    RecompExtractiveCompressor,
)

def normalize_text(text):
    """Normalize text for strict NQ exact match evaluation."""
    if not isinstance(text, str):
        text = str(text)
    # Replace non-ASCII with ?
    text = re.sub(r'[^\x00-\x7F]', '?', text)
    # Unicode normalize and lowercase
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()
    return text

def load_nq(dataset_path=None, n=20):
    """Loads PRE-PROCESSED NQ dataset and formats it. Handles multiple acceptable answers."""
    if not dataset_path or not Path(dataset_path).exists():
        raise FileNotFoundError(f"Could not find NQ dataset at {dataset_path}. Run preprocess_nq_retrieval.py first.")
        
    print(f"Loading local pre-processed NQ dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        # Load the unified JSON array
        dataset = json.load(f)

    formatted = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        
        question = item.get("question", "")
        
        # NQ has a list of acceptable answers. Extract and normalize them.
        raw_answers = item.get("answer", item.get("answers", []))
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
            
        normalized_answers = [normalize_text(ans) for ans in raw_answers]
        main_answer = normalized_answers[0] if normalized_answers else ""
        
        # Read the unified 'docs' array created by our preprocessor
        contexts = []
        for doc in item.get("docs", []):
            contexts.append([doc.get("title", ""), [doc.get("text", "")]])
            
        formatted.append({
            "question": question,
            "answer": main_answer,
            "acceptable_answers": normalized_answers, 
            "context": contexts
        })
        
    return formatted


def run(dataset_path, n):
    print(f"\nLoading NQ dataset (n={n})...")
    dataset = load_nq(dataset_path, n=n) 
    print(f"Loaded {len(dataset)} samples\n")

    output_dir = Path(project_root) / "eval_results" / "nq"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Results will be saved to: {output_dir}\n")

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    print("Initializing Reader LLM...")
    reader = RAGReader(model_name="qwen2.5:3b")

    results_table = []

    def format_metrics(name, m):
        return [
            name,
            round(m["em"], 2),
            round(m["token_f1"], 2),
            round(m["rougeL"], 2),
            round(m["disambig_f1"], 2),
            round(m["compression_ratio_chars"], 2),
            round(m.get("avg_prompt_tokens", 0), 1),
            round(m["avg_latency_sec"], 2),
        ]

    def run_and_save(name, adapter):
        print(f"\n[{name}] Running Benchmark...")
        # Evaluator already has top_k=10
        eval_result = GenerativeEvaluator(compressor=adapter, reader=reader).evaluate(dataset, top_k=10)
        
        df = pd.DataFrame(eval_result["details"])
        csv_path = output_dir / f"details_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved query details to {csv_path.name}")
        
        return eval_result["aggregate"]

    # --- 1. NoOp Baseline ---
    agg = run_and_save("NoOp", NoOpCompressor())
    results_table.append(format_metrics("NoOp", agg))

    # --- 2. EXIT Baseline ---
    exit_model = EXITCompressor(
        token=token,
        base_model="doubleyyh/exit-gemma-2b",  
        cache_dir=None 
    )
    agg = run_and_save("EXIT", ExitAdapter(exit_model))
    results_table.append(format_metrics("EXIT", agg))
    del exit_model
    gc.collect()
    torch.cuda.empty_cache()

    #--- 3. RECOMP Extractive Baseline ---
    recomp_extr = RecompExtractiveCompressor()
    agg = run_and_save("RECOMP_EXTR", RecompExtractiveAdapter(recomp_extr))
    results_table.append(format_metrics("RECOMP_EXTR", agg))
    del recomp_extr
    gc.collect()
    torch.cuda.empty_cache()

    # --- 4. LLMLingua2 Baseline ---
    llmlingua2 = LLMLingua2Compressor()
    agg = run_and_save("LLMLingua-2", LLMLingua2Adapter(llmlingua2))
    results_table.append(format_metrics("LLMLingua-2", agg))
    del llmlingua2
    gc.collect()
    torch.cuda.empty_cache()

    # --- 5. CompAct Baseline ---
    # compact = CompactCompressor(token=token)
    # agg = run_and_save("COMPACT", CompActAdapter(compact))
    # results_table.append(format_metrics("COMPACT", agg))
    # del compact
    # gc.collect()
    # torch.cuda.empty_cache()

    # --- 6. RECOMP Abstractive Baseline ---
    recomp = RECOMPAbstractiveCompressor()
    agg = run_and_save("RECOMP", RecompAdapter(recomp))
    results_table.append(format_metrics("RECOMP", agg))
    del recomp
    gc.collect()
    torch.cuda.empty_cache()

    # --- 7. REFINER Baseline ---
    # refiner = RefinerCompressor(token=token)
    # agg = run_and_save("REFINER", RefinerAdapter(refiner))
    # results_table.append(format_metrics("REFINER", agg))
    # del refiner
    # gc.collect()
    # torch.cuda.empty_cache()

    #--- 8. Hybrid Pipeline ---
    hybrid = HybridCompressor(exit_token=token)
    agg = run_and_save("HYBRID", HybridAdapter(hybrid))
    results_table.append(format_metrics("HYBRID", agg))
    del hybrid
    gc.collect()
    torch.cuda.empty_cache()

    # --- Print & Save Final Results ---
    print("\n✅ Generative Benchmark Complete!\n")
    headers = ["Compressor", "EM %", "Token F1 %", "ROUGE-L %", "Disambig-F1 %", "Compression %", "Avg Input Tokens", "Latency (s)"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    
    master_df = pd.DataFrame(results_table, columns=headers)
    master_df.to_csv(output_dir / "final_benchmark_results.csv", index=False)
    print("\n✓ Master results saved to eval_results/nq/final_benchmark_results.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Run NQ Generative Benchmark")
    parser.add_argument(
        "-n", "--num_samples", 
        type=int, 
        default=20,  # Failsafe default if -n is not passed
        help="Number of samples to evaluate (e.g., -n 500)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(dataset_path="data/nq/nq_top30_hybrid_500.json", n=args.num_samples)