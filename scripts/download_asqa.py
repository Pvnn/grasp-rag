import requests
import tarfile
import os
import json
from tqdm import tqdm

def download_alce_asqa():
    os.makedirs("data/asqa", exist_ok=True)
    tar_url = "https://huggingface.co/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar"
    tar_path = "data/asqa/ALCE-data.tar"
    
    # 1. Download the ALCE tarball (approx 1.2GB)
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
    else:
        print("Tarball already exists. Skipping download.")
    
    # 2. Extract only the ASQA DPR file
    print("\nExtracting DPR Top-100 contexts...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extract("ALCE-data/asqa_eval_dpr_top100.json", path="data/asqa")
        
    extracted_file = "data/asqa/ALCE-data/asqa_eval_dpr_top100.json"
    jsonl_file = "data/asqa/dev.jsonl"
    
    # 3. Convert to JSONL so it matches our pipeline
    print("Converting to JSONL format...")
    with open(extracted_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
    print(f"✓ ALCE ASQA Dev samples ready: {len(data)}")

if __name__ == "__main__":
    download_alce_asqa()