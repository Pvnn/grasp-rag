import json
import os
import random
import requests
import tarfile
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.hybrid_retriever import HybridRetriever

def download_alce_asqa():
    os.makedirs("data/asqa", exist_ok=True)
    tar_url = "https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar"
    tar_path = "data/asqa/ALCE-data.tar"
    
    if not os.path.exists(tar_path):
        print("Downloading ALCE ASQA with pre-retrieved DPR contexts (1.2GB)...")
        response = requests.get(tar_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(tar_path, 'wb') as file, tqdm(
            desc="Downloading ALCE-data.tar",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    
    extracted_file = "data/asqa/ALCE-data/asqa_eval_dpr_top100.json"
    if not os.path.exists(extracted_file):
        print("\nExtracting DPR Top-100 contexts...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extract("ALCE-data/asqa_eval_dpr_top100.json", path="data/asqa")
            
    return extracted_file

def preprocess_alce_asqa():
    input_file = download_alce_asqa()
    top_k = 30
    output_file = f"data/asqa/asqa_top{top_k}_hybrid_500.json"

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Randomly sample exactly 500 queries using the standard seed
    print("Sampling 500 queries with seed 42...")
    random.seed(42)
    sampled_dataset = random.sample(dataset, 500)

    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()

    print(f"Filtering {len(sampled_dataset)} queries down to Top {top_k} documents...")
    reduced_dataset = []

    for item in tqdm(sampled_dataset, desc="Processing ASQA"):
        query = item.get("ambiguous_question", item.get("question", ""))
        
        # 1. Extract the 100 docs
        raw_docs = []
        for i, doc in enumerate(item.get("docs", [])):
            raw_docs.append({
                "id": i,
                "title": doc.get("title", "Wikipedia"),
                "text": doc.get("text", "")
            })

        # 2. Re-rank with HybridRetriever
        if raw_docs:
            retriever.index_documents(raw_docs)
            retrieved_items = retriever.retrieve(query, top_k=top_k)
            
            # 3. Save only the Top-K back into the item
            top_docs = []
            for doc_dict, score in retrieved_items:
                top_docs.append({
                    "title": doc_dict["title"],
                    "text": doc_dict["text"],
                    "retrieval_score": score
                })
            item["docs"] = top_docs
        else:
            item["docs"] = []

        reduced_dataset.append(item)

    # Save the new lightweight dataset
    print(f"Saving reduced dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reduced_dataset, f, indent=4)
        
    print("✅ ASQA Preprocessing complete! You can now run your eval script without the retriever.")

if __name__ == "__main__":
    preprocess_alce_asqa()