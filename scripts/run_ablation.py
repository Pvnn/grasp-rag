import sys
import os
import gc
import torch
import pandas as pd
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load env variables BEFORE importing any HF libraries so cache paths apply
load_dotenv()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tabulate import tabulate
from src.eval.eval_pipeline import GenerativeEvaluator
from src.generation.reader import RAGReader
from src.compression.hybrid_compressor import HybridCompressor
from src.eval.adapters import NoOpCompressor

# We use the dataset loader functions from the existing eval scripts
from src.eval.eval_nq import load_nq
from src.eval.eval_hotpotqa import load_hotpotqa

class AblationAdapter:
    """
    A custom adapter that wraps the HybridCompressor to inject specific
    hyperparameters for the ablation study at runtime.
    """
    def __init__(self, compressor: HybridCompressor, config: dict):
        self.compressor = compressor
        self.config = config

    def compress(self, query: str, docs: list) -> dict:
        # Dynamically set graph parameters inside EP-EXIT before running
        if hasattr(self.compressor, "exit") and self.compressor.exit is not None:
            self.compressor.exit.locality_window = self.config.get("w", 2)
            self.compressor.exit.similarity_threshold = self.config.get("delta", 0.45)
            
        result = self.compressor.compress(
            query=query,
            context=docs,      
            use_coarse=self.config.get("use_coarse", True),
            use_fine=self.config.get("use_fine", True),
            coarse_ratio=self.config.get("coarse_ratio", 0.8),
            fine_threshold=self.config.get("fine_threshold", 0.5)
        )
        return {"compressed_docs": result.get("compressed_docs", [])}


def format_metrics(name, dataset_name, m):
    return {
        "Config Name": name,
        "Dataset": dataset_name,
        "EM %": round(m["em"], 2),
        "Token F1 %": round(m["token_f1"], 2),
        "ROUGE-L %": round(m["rougeL"], 2),
        "Disambig-F1 %": round(m["disambig_f1"], 2),
        "Compression %": round(m["compression_ratio_chars"], 2),
        "Avg Input Tokens": round(m.get("avg_prompt_tokens", 0), 1),
        "Latency (s)": round(m["avg_latency_sec"], 2)
    }

def run_ablation(n=300):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN not found in .env. Exiting.")
        return

    output_dir = project_root / "src" / "eval_results" / "ablation_study"
    output_dir.mkdir(exist_ok=True, parents=True)
    master_csv = output_dir / "ablation_results.csv"

    # Load datasets
    print(f"\nLoading Datasets (N={n})...")
    nq_data = load_nq("data/nq/nq_top30_hybrid_500.json", n=n)
    hotpotqa_data = load_hotpotqa("data/hotpotqa/hotpotqa_top30_hybrid_500.json", n=n)

    # Initialize models ONCE for GPU efficiency
    print("\nInitializing Reader LLM (qwen2.5:3b)...")
    reader = RAGReader(model_name="qwen2.5:3b")

    print("Initializing HybridCompressor (Flan-T5-Small + Gemma-2B)...")
    hybrid_compressor = HybridCompressor(exit_token=token)

    # -------------------------------------------------------------------------
    # ABLATION CONFIGURATIONS
    # -------------------------------------------------------------------------
    # Format: (Config Name, Dataset Name, Dataset Object, Adapter/Dict)
    configs = [
        # --- Part 1: Macro-Ablation (NQ) ---
        ("1_Macro_NQ_NoOp", "Natural Questions", nq_data, "NoOp"),
        ("1_Macro_NQ_CoarseOnly", "Natural Questions", nq_data, {"use_coarse": True, "use_fine": False}),
        ("1_Macro_NQ_VanillaEXIT", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 0}),
        ("1_Macro_NQ_EPEXIT", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 2}),
        ("1_Macro_NQ_GRASP", "Natural Questions", nq_data, {"use_coarse": True, "use_fine": True, "w": 2}),

        # --- Part 1: Macro-Ablation (HotpotQA) ---
        ("1_Macro_HQA_NoOp", "HotpotQA", hotpotqa_data, "NoOp"),
        ("1_Macro_HQA_CoarseOnly", "HotpotQA", hotpotqa_data, {"use_coarse": True, "use_fine": False}),
        ("1_Macro_HQA_VanillaEXIT", "HotpotQA", hotpotqa_data, {"use_coarse": False, "use_fine": True, "w": 0}),
        ("1_Macro_HQA_EPEXIT", "HotpotQA", hotpotqa_data, {"use_coarse": False, "use_fine": True, "w": 2}),
        ("1_Macro_HQA_GRASP", "HotpotQA", hotpotqa_data, {"use_coarse": True, "use_fine": True, "w": 2}),

        # --- Part 2A: Micro-Ablation Graph Delta (NQ) [EP-EXIT Only] ---
        # Delta=0.45 is already covered by 1_Macro_NQ_EPEXIT (default is 0.45)
        ("2A_Delta_0.2", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 2, "delta": 0.2}),
        ("2A_Delta_0.35", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 2, "delta": 0.35}),
        ("2A_Delta_0.6", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 2, "delta": 0.6}),
        ("2A_Delta_0.8", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 2, "delta": 0.8}),

        # --- Part 2B: Micro-Ablation Graph Window (NQ) [EP-EXIT Only] ---
        # w=0 covered by 1_Macro_NQ_VanillaEXIT, w=2 covered by 1_Macro_NQ_EPEXIT
        ("2B_Window_1", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 1, "delta": 0.45}),
        ("2B_Window_3", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 3, "delta": 0.45}),
        ("2B_Window_5", "Natural Questions", nq_data, {"use_coarse": False, "use_fine": True, "w": 5, "delta": 0.45}),

        # --- Part 3: Load Balancing Coarse vs Fine (NQ) ---
        # Balanced (r=0.8, tau=0.5) is implicitly covered by 1_Macro_NQ_GRASP (defaults).
        ("3_LoadBal_AggressiveCoarse", "Natural Questions", nq_data, {"use_coarse": True, "use_fine": True, "w": 2, "coarse_ratio": 0.5, "fine_threshold": 0.4}),
        ("3_LoadBal_PassiveCoarse", "Natural Questions", nq_data, {"use_coarse": True, "use_fine": True, "w": 2, "coarse_ratio": 1.0, "fine_threshold": 0.6}),
    ]

    # Load existing results to support incremental resume
    if master_csv.exists():
        results_df = pd.read_csv(master_csv)
        completed_configs = results_df["Config Name"].tolist()
        print(f"\nResuming from existing {master_csv.name}. {len(completed_configs)} configs already done.")
    else:
        results_df = pd.DataFrame()
        completed_configs = []

    for name, dataset_name, dataset, config_or_string in configs:
        if name in completed_configs:
            print(f"Skipping {name}, already completed.")
            continue

        print(f"\n=======================================================")
        print(f" Running Config: {name}")
        print(f"=======================================================")

        if config_or_string == "NoOp":
            adapter = NoOpCompressor()
        else:
            adapter = AblationAdapter(hybrid_compressor, config_or_string)

        evaluator = GenerativeEvaluator(compressor=adapter, reader=reader)
        
        # Run evaluation
        eval_result = evaluator.evaluate(dataset, top_k=10)
        
        # Save details CSV
        details_df = pd.DataFrame(eval_result["details"])
        details_path = output_dir / f"details_{name}.csv"
        details_df.to_csv(details_path, index=False)
        
        # Format aggregate metrics and append to master dataframe
        agg_metrics = format_metrics(name, dataset_name, eval_result["aggregate"])
        
        if results_df.empty:
            results_df = pd.DataFrame([agg_metrics])
        else:
            results_df = pd.concat([results_df, pd.DataFrame([agg_metrics])], ignore_index=True)
            
        # Save master CSV incrementally
        results_df.to_csv(master_csv, index=False)
        print(f"✓ Saved results for {name} to {master_csv.name}")

        # Cleanup Memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\nAll ablation experiments completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRASP Ablation Study")
    parser.add_argument("-n", "--num_samples", type=int, default=300, help="Number of samples per config (default: 300)")
    args = parser.parse_args()
    
    run_ablation(n=args.num_samples)
