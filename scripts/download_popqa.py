from datasets import load_dataset

popqa = load_dataset("akariasai/PopQA", split="test")
popqa.save_to_disk("data/popqa/test")

print("Test samples:", len(popqa))