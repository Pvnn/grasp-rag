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

class SelectiveContextAdapter:
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query: str, docs):
        result = self.compressor.compress(query, docs)
        return {"compressed_docs": result}


class FILCoAdapter:
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query: str, docs):
        result = self.compressor.compress(query, docs)
        return {"compressed_docs": result}

class LLMLinguaAdapter:
    """
    Adapter for LLMLingua (original).
    Paper: Jiang et al., EMNLP 2023 — https://arxiv.org/abs/2310.05736
    """
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query: str, docs) -> dict:
        result = self.compressor.compress(query, docs)
        return {"compressed_docs": result}


class LongLLMLinguaAdapter:
    """
    Adapter for LongLLMLingua.
    Paper: Jiang et al., ACL 2024 — https://arxiv.org/abs/2310.06839
    """
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query: str, docs) -> dict:
        result = self.compressor.compress(query, docs)
        return {"compressed_docs": result}


class JinaRerankerAdapter:
    """
    Adapter for JinaAI Reranker used as compressor.
    Model: jinaai/jina-reranker-v2-base-multilingual
    """
    def __init__(self, compressor):
        self.compressor = compressor

    def compress(self, query: str, docs) -> dict:
        result = self.compressor.compress(query, docs)
        return {"compressed_docs": result}