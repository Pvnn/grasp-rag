import os
from datasets import load_dataset

# Create directory
os.makedirs("data/nq", exist_ok=True)

print("Downloading NQ Open with pre-retrieved DPR contexts...")
# We use the xfact/nq-dpr dataset which includes the top 100 Wikipedia passages per question
nq_dev = load_dataset("xfact/nq-dpr", split="validation")

# Export directly to JSONL format
print("Saving to JSONL...")
nq_dev.to_json("data/nq/dev.jsonl")

print(f"✓ NQ Dev samples ready: {len(nq_dev)}")