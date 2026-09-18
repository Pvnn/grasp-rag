import json
import os
import random
from tqdm import tqdm
import sys
from pathlib import Path

import wikipedia

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from src.retrieval.hybrid_retriever import HybridRetriever


def fetch_wiki_chunks(title: str, max_chunks: int = 10, chunk_size: int = 200) -> list:
    """Fetch and chunk a Wikipedia page. Returns empty list on any failure."""
    try:
        page = wikipedia.page(title, auto_suggest=False, preload=False)
        words = page.content.split()
        if len(words) < 50:   # too short to be useful
            return []
        return [
            {"title": title, "text": " ".join(words[i:i + chunk_size])}
            for i in range(0, min(len(words), chunk_size * max_chunks), chunk_size)
        ]
    except wikipedia.exceptions.DisambiguationError as e:
        # Try the first disambiguation option
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            words = page.content.split()
            if len(words) < 50:
                return []
            return [
                {"title": e.options[0], "text": " ".join(words[i:i + chunk_size])}
                for i in range(0, min(len(words), chunk_size * max_chunks), chunk_size)
            ]
        except Exception:
            return []
    except Exception:
        return []


def preprocess_popqa():
    os.makedirs("data/popqa", exist_ok=True)
    top_k = 30
    output_file = f"data/popqa/popqa_top{top_k}_hybrid_500.json"

    print("Downloading PopQA from HuggingFace...")
    hf_dataset = load_dataset("akariasai/PopQA", split="test")
    dataset_list = list(hf_dataset)

    # Shuffle with fixed seed so results are reproducible
    random.seed(42)
    random.shuffle(dataset_list)

    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()

    reduced_dataset = []
    skipped = 0

    # Iterate through the full shuffled list until we have 500 good items
    for item in tqdm(dataset_list, desc="Processing PopQA"):
        if len(reduced_dataset) >= 500:
            break

        query   = item["question"]
        answer  = item["obj"]

        raw_docs = []

        # Subject entity page — primary source
        subj_title = item.get("s_wiki_title", "")
        if subj_title:
            raw_docs.extend(fetch_wiki_chunks(subj_title, max_chunks=10))

        # Object entity page — secondary context
        obj_title = item.get("o_wiki_title", "")
        if obj_title and obj_title != subj_title:
            raw_docs.extend(fetch_wiki_chunks(obj_title, max_chunks=5))

        # Skip items where we couldn't get any content
        if not raw_docs:
            skipped += 1
            continue

        # Assign sequential IDs
        for i, doc in enumerate(raw_docs):
            doc["id"] = i

        retriever.index_documents(raw_docs)
        retrieved_items = retriever.retrieve(query, top_k=top_k)

        top_docs = [
            {"title": doc_dict["title"], "text": doc_dict["text"], "retrieval_score": score}
            for doc_dict, score in retrieved_items
        ]

        # Skip if retriever returned nothing useful
        if not top_docs:
            skipped += 1
            continue

        acceptable = item.get("possible_answers", [answer])
        if isinstance(acceptable, str):
            try:
                acceptable = json.loads(acceptable)
            except Exception:
                acceptable = [acceptable]

        reduced_dataset.append({
            "question": query,
            "answer": answer,
            "acceptable_answers": acceptable,
            "docs": top_docs
        })

    print(f"Collected {len(reduced_dataset)} samples, skipped {skipped} with no retrievable content.")

    if len(reduced_dataset) < 500:
        print(f"⚠️  Warning: only {len(reduced_dataset)} samples available with docs — saving what we have.")

    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reduced_dataset, f, indent=4)

    print("✅ PopQA Preprocessing complete!")

if __name__ == "__main__":
    preprocess_popqa()