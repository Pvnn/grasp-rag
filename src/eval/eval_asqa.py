import sys
from pathlib import Path
import gc
import torch
import os
import pandas as pd 
from dotenv import load_dotenv
import argparse
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from tabulate import tabulate

from src.eval.eval_pipeline import GenerativeEvaluator
from src.eval.adapters import (
    NoOpCompressor, ExitAdapter, HybridAdapter,
    RefinerAdapter, RecompAdapter, LLMLingua2Adapter,
    CompActAdapter, RecompExtractiveAdapter
)

# Import Models
from src.compression.hybrid_compressor import HybridCompressor
from src.compression.baselines import (
    EXITCompressor,
    RefinerCompressor,
    RECOMPAbstractiveCompressor,
    LLMLingua2Compressor,
    CompactCompressor,
    RecompExtractiveCompressor,
)
from src.generation.reader import RAGReader


def load_asqa(dataset_path=None, n=20):
    """Loads PRE-PROCESSED ALCE-ASQA dataset and formats it for the evaluator."""
    dataset = []
    
    if dataset_path and Path(dataset_path).exists():
        print(f"Loading local pre-processed ASQA dataset from {dataset_path}...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            # FIX: Our pre-processor saves a standard JSON array, so we load the whole file at once
            dataset = json.load(f)
    else:
        raise FileNotFoundError(f"Could not find dataset at {dataset_path}. Please run the preprocess script.")
        
    formatted = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        
        # ALCE stores the ambiguous question here
        question = item.get("ambiguous_question", item.get("question", ""))
        
        short_answers = []
        if "qa_pairs" in item:
            for qa in item["qa_pairs"]:
                if "short_answers" in qa:
                    short_answers.extend(qa["short_answers"])
                    
        long_answers = []
        if "annotations" in item:
            for ann in item["annotations"]:
                if "long_answer" in ann:
                    long_answers.append(ann["long_answer"])
        
        # Extract the Top-10 DPR passages saved by our preprocessor
        contexts = []
        for doc in item.get("docs", []):
            title = doc.get("title", "Wikipedia")
            # Grab the actual text payload
            text = doc.get("text", "") 
            if text.strip():
                # Format expected by GenerativeEvaluator: [title, [sentences]]
                contexts.append([title, [text.strip()]])
            
        formatted.append({
            "question": question,
            "answer": long_answers[0] if long_answers else "",
            # For ASQA, we use the combined short_answers for the Exact Match metric
            "acceptable_answers": short_answers, 
            "context": contexts  
        })
        
    return formatted


def run(dataset_path, n):
    print(f"\nLoading ASQA dataset (n={n})...")
    dataset = load_asqa(dataset_path, n=n) 
    print(f"Loaded {len(dataset)} samples\n")

    output_dir = Path(project_root) / "eval_results" / "asqa"
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
        # FIX: Pass top_k=10 to the evaluator so it maps the Excel files perfectly
        eval_result = GenerativeEvaluator(compressor=adapter, reader=reader).evaluate(dataset, top_k=10)
        
        df = pd.DataFrame(eval_result["details"])
        csv_path = output_dir / f"details_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved query details to {csv_path.name}")
        
        return eval_result["aggregate"]
    
    # --- 1. NoOp Baseline ---
    agg = run_and_save("NoOp", NoOpCompressor())
    results_table.append(format_metrics("NoOp", agg))

    # --- 2. EXIT Baseline  ---
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

    # --- 3. RECOMP Extractive Baseline ---
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

    # --- 6. RECOMP Baseline ---
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

    # --- 8. Hybrid Pipeline ---
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
    print("\n✓ Master results saved to eval_results/asqa/final_benchmark_results.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Run ASQA Generative Benchmark")
    parser.add_argument(
        "-n", "--num_samples", 
        type=int, 
        default=20,  # Failsafe default if -n is not passed
        help="Number of samples to evaluate (e.g., -n 500)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(dataset_path="data/asqa/asqa_top30_hybrid_500.json", n=args.num_samples)