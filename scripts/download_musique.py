from datasets import load_dataset

musique_dev = load_dataset("MemoryAsModality/MuSiQue", split="validation")
musique_dev.save_to_disk("data/musique/dev")

musique_train = load_dataset("MemoryAsModality/MuSiQue", split="train")
musique_train.save_to_disk("data/musique/train")

print("Dev samples:", len(musique_dev))
print("Train samples:", len(musique_train))