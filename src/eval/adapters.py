"""
Adapters for wrapping various compression baselines into a unified interface
for the GenerativeEvaluator.
"""
from typing import List
from src.eval.interfaces import SearchResult

class NoOpCompressor:
    def compress(self, query, docs):
        return {"compressed_docs": docs}

class ExitAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
 
    def compress(self, query: str, docs: List[SearchResult]) -> dict:
        # Pass all docs at once — the compressor concatenates them internally,
        # splits into sentences across the full corpus, then batch-classifies.
        # Never call this in a per-doc loop: that destroys cross-doc batching
        # and reverts to the slow serial behavior we fixed.
        result = self.compressor.compress(query, docs)
        return {"compressed_docs": result}

class QuitoAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        compressed_docs = []
        for doc in docs:
            sentences = [s.strip() + "." for s in doc.text.split(".") if s.strip()]
            result = self.compressor.compress(query=query, sentences=sentences, compression_ratio=0.5)
            text = " ".join(result["filtered_sentences"])
            compressed_docs.append(SearchResult(evi_id=doc.evi_id, docid=doc.docid, title=doc.title, text=text))
        return {"compressed_docs": compressed_docs}

class HybridAdapter:
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query: str, docs: List[SearchResult]) -> dict:
        result = self.compressor.compress(
            query=query,
            context=docs,      
            use_coarse=True,
            use_fine=True
        )
        return {"compressed_docs": result["compressed_docs"]}

class RefinerAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class RecompAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class LLMLingua2Adapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class CompActAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}

class RecompExtractiveAdapter:
    def __init__(self, compressor):
        self.compressor = compressor
    def compress(self, query, docs):
        return {"compressed_docs": self.compressor.compress(query, docs)}