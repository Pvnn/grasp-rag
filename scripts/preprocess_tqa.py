import json
import os
from tqdm import tqdm
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from src.retrieval.hybrid_retriever import HybridRetriever

def chunk_text(text, chunk_size=150):
    """Splits massive Wikipedia/Web pages into smaller, RAG-friendly paragraph chunks."""
    words = text.replace('\n', ' ').split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def preprocess_tqa():
    os.makedirs("data/tqa", exist_ok=True)
    top_k = 30 
    output_file = f"data/tqa/tqa_top{top_k}_hybrid_500.json" 

    print("Streaming TriviaQA (rc) directly from HuggingFace...")
    hf_dataset = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
    
    # Grab exactly the first 500 records from the stream instantly
    print("Fetching 500 queries...")
    sampled_dataset = list(hf_dataset.take(500))

    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()

    print(f"Filtering {len(sampled_dataset)} queries and ordering Top {top_k} documents...")
    reduced_dataset = []

    for item in tqdm(sampled_dataset, desc="Processing TriviaQA"):
        query = item["question"]
        
        answers = item.get("answer", {}).get("normalized_aliases", [])
        if not answers:
            answers = [item.get("answer", {}).get("value", "")]
            
        raw_docs = []
        doc_id = 0
        
        if "entity_pages" in item:
            for t, c in zip(item["entity_pages"].get("title", []), item["entity_pages"].get("wiki_context", [])):
                if c.strip():
                    paragraphs = chunk_text(c)
                    for para in paragraphs:
                        raw_docs.append({"id": doc_id, "title": t, "text": para})
                        doc_id += 1
                    
        if "search_results" in item:
            for t, c in zip(item["search_results"].get("title", []), item["search_results"].get("search_context", [])):
                if c.strip():
                    paragraphs = chunk_text(c)
                    for para in paragraphs:
                        raw_docs.append({"id": doc_id, "title": t, "text": para})
                        doc_id += 1

        if raw_docs:
            retriever.index_documents(raw_docs)
            retrieved_items = retriever.retrieve(query, top_k=top_k)
            
            top_docs = [{"title": d["title"], "text": d["text"]} for d, s in retrieved_items]
        else:
            top_docs = []

        reduced_dataset.append({
            "question": query,
            "acceptable_answers": answers,
            "docs": top_docs
        })

    print(f"Saving reduced dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reduced_dataset, f, indent=4)
        
    print("✅ TriviaQA Preprocessing complete!")

if __name__ == "__main__":
    preprocess_tqa()