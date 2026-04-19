import sys
from pathlib import Path
import gc
import torch
import os
import pandas as pd 
import json
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

def load_tqa(dataset_path, n=20):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    formatted = []
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        contexts = [[doc.get("title", ""), [doc.get("text", "")]] for doc in item.get("docs", [])]
        
        acceptable_answers = item.get("acceptable_answers", [])
        
        formatted.append({
            "question": item.get("question", ""),
            "answer": str(acceptable_answers),
            "acceptable_answers": acceptable_answers,
            "context": contexts
        })
    return formatted

def run(dataset_path, n):
    print(f"\nLoading TriviaQA dataset (n={n})...")
    dataset = load_tqa(dataset_path, n=n) 
    print(f"Loaded {len(dataset)} samples\n")

    output_dir = Path(project_root) / "eval_results" / "tqa"
    output_dir.mkdir(exist_ok=True, parents=True)

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    reader = RAGReader(model_name="qwen2.5:3b")
    results_table = []

    def format_metrics(name, m):
        return [
            name, round(m["em"], 2), round(m["token_f1"], 2), round(m["rougeL"], 2),
            round(m["disambig_f1"], 2), round(m["compression_ratio_chars"], 2),
            round(m.get("avg_prompt_tokens", 0), 1), round(m["avg_latency_sec"], 2),
        ]

    def run_and_save(name, adapter):
        print(f"\n[{name}] Running Benchmark...")
        eval_result = GenerativeEvaluator(compressor=adapter, reader=reader).evaluate(dataset, top_k=10)
        df = pd.DataFrame(eval_result["details"])
        df.to_csv(output_dir / f"details_{name}.csv", index=False)
        return eval_result["aggregate"]

    # 1. NoOp Baseline
    agg = run_and_save("NoOp", NoOpCompressor())
    results_table.append(format_metrics("NoOp", agg))

    # 2. EXIT Baseline
    exit_model = EXITCompressor(token=token, base_model="doubleyyh/exit-gemma-2b", cache_dir=None)
    agg = run_and_save("EXIT", ExitAdapter(exit_model))
    results_table.append(format_metrics("EXIT", agg))
    del exit_model; gc.collect(); torch.cuda.empty_cache()

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

    print("\n✅ Generative Benchmark Complete!\n")
    headers = ["Compressor", "EM %", "Token F1 %", "ROUGE-L %", "Disambig-F1 %", "Compression %", "Avg Input Tokens", "Latency (s)"]
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    pd.DataFrame(results_table, columns=headers).to_csv(output_dir / "final_benchmark_results.csv", index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Run TQA Generative Benchmark")
    parser.add_argument(
        "-n", "--num_samples", 
        type=int, 
        default=20,  # Failsafe default if -n is not passed
        help="Number of samples to evaluate (e.g., -n 500)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(dataset_path="data/tqa/tqa_top30_hybrid_500.json", n=args.num_samples)