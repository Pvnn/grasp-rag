import json
from pathlib import Path

# Static curated docs for the JWST playground
JWST_DOCS = [
    {"title": "doc1", "text": "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy. Its high-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope. The primary scientific instruments on board include the Near-Infrared Camera (NIRCam) and the Mid-Infrared Instrument (MIRI). These tools are essential for studying the formation of early galaxies. Additionally, the telescope requires a massive sunshield to keep its instruments cold."},
    {"title": "doc2", "text": "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in operation. It features a 2.4-meter mirror and observes primarily in the visible and ultraviolet spectra. Hubble has recorded some of the most detailed visible-light images ever, allowing a deep view into space. It does not possess the advanced mid-infrared capabilities of newer observatories. Astronauts have visited Hubble multiple times to repair and upgrade its systems."},
    {"title": "doc3", "text": "One of the major goals of modern astronomy is the study of exoplanets and their atmospheres. JWST uses its Near-Infrared Spectrograph (NIRSpec) to analyze the chemical composition of exoplanetary atmospheres. By observing the transit of a planet across its host star, scientists can detect signatures of water and carbon dioxide. This method of transit spectroscopy is highly dependent on the telescope's stable orbit. Future missions will continue this search for habitable worlds."},
    {"title": "doc4", "text": "JWST was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana. It took a month to reach its destination, the Sun-Earth L2 Lagrange point. This orbit allows the telescope to stay in line with Earth as it moves around the Sun. The L2 point is approximately 1.5 million kilometers away from our planet. Operations and data processing are managed by the Space Telescope Science Institute."}
]

def load_demo_datasets(max_queries=50):
    """
    Loads the static JWST playground AND dynamically loads the 
    pre-processed benchmark datasets into the UI.
    """
    base_dir = Path("data")
    
    # 1. Initialize with the custom JWST dataset
    demo_data = {
        "jwst": {
            "name": "James Webb Space Telescope (Curated)",
            "queries": [
                "What are the specific instruments on the James Webb Space Telescope used for observation?",
                "Where was the JWST launched from and what is its destination?",
                "How does JWST study exoplanets?"
            ],
            "global_documents": JWST_DOCS,
            "query_documents": {} 
        }
    }
    
    # 2. Map the dynamic benchmark datasets
    dataset_files = {
        "2wiki": ("2WikiMultihopQA", "2wiki/2wiki_top30_hybrid_500.json"),
        "tqa": ("TriviaQA", "tqa/tqa_top30_hybrid_500.json"),
        "hotpotqa": ("HotpotQA", "hotpotqa/hotpotqa_top30_hybrid_500.json"),
        "nq": ("Natural Questions", "nq/nq_top30_hybrid_500.json"),
        "asqa": ("ASQA", "asqa/asqa_top30_hybrid_500.json")
    }

    # 3. Load JSON files
    for ds_id, (name, path_str) in dataset_files.items():
        full_path = base_dir / path_str
        
        if not full_path.exists():
            print(f"⚠️ Skipping {name}: Could not find {full_path}")
            continue

        print(f"Loading {name} for UI...")
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        queries = []
        query_docs = {}

        # Limit to max_queries to keep the React UI snappy
        for item in data[:max_queries]:
            q = item.get("question", "")
            docs = item.get("docs", [])
            
            # Format docs exactly how the retriever expects them
            formatted_docs = [
                {
                    "title": d.get("title", f"Doc {i+1}"), 
                    "text": d.get("text", "")
                } 
                for i, d in enumerate(docs)
            ]
            
            queries.append(q)
            query_docs[q] = formatted_docs

        # Add the loaded dataset to the master dictionary
        demo_data[ds_id] = {
            "name": name,
            "queries": queries,
            "query_documents": query_docs,
            "global_documents": [] # We don't need a global fallback for these datasets
        }
    
    return demo_data