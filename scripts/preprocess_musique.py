import json
import os
import random
from tqdm import tqdm
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from src.retrieval.hybrid_retriever import HybridRetriever

def preprocess_musique():
    os.makedirs("data/musique", exist_ok=True)
    top_k = 30
    output_file = f"data/musique/musique_top{top_k}_hybrid_500.json"
    if os.path.exists(output_file):
        print(f"Skipping {output_file} as it already exists.")
        return


    print("Downloading MuSiQue from HuggingFace...")
    hf_dataset = load_dataset("MemoryAsModality/MuSiQue", split="validation")

    print("Sampling 500 queries with seed 42...")
    dataset_list = list(hf_dataset)
    random.seed(42)
    sampled_dataset = random.sample(dataset_list, min(500, len(dataset_list)))

    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()

    print(f"Processing {len(sampled_dataset)} queries, Top {top_k} documents...")
    reduced_dataset = []

    for item in tqdm(sampled_dataset, desc="Processing MuSiQue"):
        query = item["question"]
        answer = item["answer"]
        # MuSiQue provides supporting paragraphs directly
        raw_docs = []
        for i, doc_text in enumerate(item["documents"]):
          raw_docs.append({
              "id": i,
              "title": f"Doc {i}",
              "text": doc_text
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

        reduced_dataset.append({
            "question": query,
            "answer": answer,
            "acceptable_answers": item.get("answer_aliases", []) or [answer],
            "docs": top_docs
        })

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reduced_dataset, f, indent=4)

    print("✅ MuSiQue Preprocessing complete!")

if __name__ == "__main__":
    preprocess_musique()