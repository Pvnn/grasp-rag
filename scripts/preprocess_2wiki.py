import json
import os
import random
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from src.retrieval.hybrid_retriever import HybridRetriever

def preprocess_2wiki():
    os.makedirs("data/2wiki", exist_ok=True)
    top_k = 30
    output_file = f"data/2wiki/2wiki_top{top_k}_hybrid_500.json"

    # Using the stable, active mirror formatted identically to HotpotQA
    print("Downloading 2WikiMultihopQA from HuggingFace (framolfese mirror)...")
    hf_dataset = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    
    # Randomly sample exactly 500 queries using the standard seed
    print("Sampling 500 queries with seed 42...")
    dataset_list = list(hf_dataset)
    random.seed(42)
    sampled_dataset = random.sample(dataset_list, 500)

    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()

    print(f"Filtering {len(sampled_dataset)} queries and ordering Top {top_k} documents...")
    reduced_dataset = []

    for item in tqdm(sampled_dataset, desc="Processing 2WikiMultihopQA"):
        query = item["question"]
        answer = item["answer"]
        
        raw_docs = []
        # framolfese mirror uses the exact same layout as HotpotQA
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        
        for i, (title, sents) in enumerate(zip(titles, sentences)):
            raw_docs.append({
                "id": i,
                "title": title,
                "text": " ".join(sents)
            })

        if raw_docs:
            retriever.index_documents(raw_docs)
            retrieved_items = retriever.retrieve(query, top_k=top_k)
            
            top_docs = []
            for doc_dict, score in retrieved_items:
                top_docs.append({
                    "title": doc_dict["title"],
                    "text": doc_dict["text"],
                    "retrieval_score": score
                })
        else:
            top_docs = []

        # Save to the unified format
        reduced_dataset.append({
            "question": query,
            "answer": answer,
            "acceptable_answers": [answer],
            "docs": top_docs
        })

    print(f"Saving reduced dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reduced_dataset, f, indent=4)
        
    print("✅ 2WikiMultihopQA Preprocessing complete!")

if __name__ == "__main__":
    preprocess_2wiki()