from dataclasses import dataclass
from typing import List
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch

from src.compression.exit_baseline import ExitBaselineCompressor

@dataclass
class EvidenceUnit:
    sentences: List[str]
    indices: List[int] 
    start_idx: int
    end_idx: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

class EPExitCompressor:
    def __init__(
        self,
        token,
        model_name="doubleyyh/exit-gemma-2b",
        threshold=0.4,
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.45,
        locality_window=2
    ):
        print("Initializing EPExitCompressor...")
        self.similarity_threshold = similarity_threshold
        self.locality_window = locality_window
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        if self.device == "cuda":
            self.embedder.half()
        self.exit = ExitBaselineCompressor(
            token=token, 
            model_name=model_name, 
            threshold=threshold
        )
        print("✓ EP-EXIT initialized\n")

    def decompose_sentences(self, text):
        return self.exit.decompose_sentences(text)

    def classify_sentence(self, query, sentence, document):
        return self.exit.classify_sentence(query, sentence, document)

    def build_similarity_graph(self, sentences, parent_contexts=None):
        embeddings = self.embedder.encode(
            sentences, 
            convert_to_tensor=True,
            batch_size=64, 
            normalize_embeddings=True 
        )
        
        sim_matrix = util.cos_sim(embeddings, embeddings)
        sim_matrix_cpu = sim_matrix.cpu().numpy()
        
        n = len(sentences)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            limit = min(i + 1 + self.locality_window, n)
            for j in range(i + 1, limit):
                if sim_matrix_cpu[i][j] >= self.similarity_threshold:
                    if parent_contexts and parent_contexts[i] != parent_contexts[j]:
                        continue
                    G.add_edge(i, j)
        return G

    def extract_evidence_units(self, graph, sentences):
        units = []
        for component in nx.connected_components(graph):
            idxs = sorted(component)
            unit_sentences = [sentences[i] for i in idxs]
            units.append(
                EvidenceUnit(
                    sentences=unit_sentences,
                    indices=idxs, 
                    start_idx=idxs[0],
                    end_idx=idxs[-1],
                )
            )
        units.sort(key=lambda u: u.start_idx)
        return units

    def compress(self, query: str, document: str) -> str:
        sentences = self.decompose_sentences(document)
        if not sentences:
            return ""

        graph = self.build_similarity_graph(sentences)
        units = self.extract_evidence_units(graph, sentences)

        # Batch classify all evidence units in one shot
        unit_texts = [unit.text for unit in units]
        queries_list = [query] * len(unit_texts)

        kept_indices = set()
        for i in range(0, len(unit_texts), self.exit.batch_size):
            batch_texts   = unit_texts[i : i + self.exit.batch_size]
            batch_queries = queries_list[i : i + self.exit.batch_size]
            batch_units   = units[i : i + self.exit.batch_size]
            yes_probs, _  = self.exit._predict_batch(batch_queries, batch_texts, document)
            for unit, score in zip(batch_units, yes_probs):
                if score > self.threshold:
                    kept_indices.update(unit.indices)

        return " ".join(sentences[i] for i in sorted(kept_indices))

    # Update this method inside your EPExitCompressor class
    def compress_with_stats(self, query, sentences, parent_contexts=None):
        if not sentences:
            return {
                "compressed_text": "", "original_length": 0, "compressed_length": 0,
                "compression_ratio": 0.0, "sentences_kept": 0, "sentences_total": 0,
                "evidence_units_total": 0, "evidence_units_kept_count": 0,
                "evidence_units_removed_count": 0, "total_tokens_consumed": 0,
                "all_units": [], "kept_units": [], "removed_units": []
            }

        graph = self.build_similarity_graph(sentences, parent_contexts=parent_contexts)
        units = self.extract_evidence_units(graph, sentences)

        unit_texts = [unit.text for unit in units]
        queries_list = [query] * len(unit_texts)
        
        # Grab the specific parent context for the first sentence of each EvidenceUnit
        if parent_contexts:
            unit_contexts = [parent_contexts[unit.indices[0]] for unit in units]
        else:
            unit_contexts = [" ".join(sentences)] * len(units)

        kept_indices = set()
        all_units_info, kept_units_info, removed_units_info = [], [], []
        total_tokens = 0

        for i in range(0, len(unit_texts), self.exit.batch_size):
            batch_texts = unit_texts[i : i + self.exit.batch_size]
            batch_queries = queries_list[i : i + self.exit.batch_size]
            batch_contexts = unit_contexts[i : i + self.exit.batch_size]
            batch_units = units[i : i + self.exit.batch_size]

            # Pass the list of mapped contexts!
            yes_probs, batch_tokens = self.exit._predict_batch(
                batch_queries, batch_contexts, batch_texts
            )
            total_tokens += batch_tokens

            for unit, score in zip(batch_units, yes_probs):
                kept = score > self.threshold
                info = {
                    "text": unit.text, "sentences": unit.sentences,
                    "indices": unit.indices, "start_idx": unit.start_idx,
                    "end_idx": unit.end_idx, "score": score
                }
                all_units_info.append(info)
                if kept:
                    kept_units_info.append(info)
                    kept_indices.update(unit.indices)
                else:
                    removed_units_info.append(info)

        compressed_text = " ".join(sentences[i] for i in sorted(kept_indices))
        return {
            "compressed_text": compressed_text,
            "original_length": sum(len(c) for c in set(parent_contexts)) if parent_contexts else 0,
            "compressed_length": len(compressed_text),
            "compression_ratio": len(compressed_text) / sum(len(c) for c in set(parent_contexts)) if parent_contexts else 0.0,
            "sentences_kept": len(kept_indices),
            "sentences_total": len(sentences),
            "evidence_units_total": len(units),
            "evidence_units_kept_count": len(kept_units_info),
            "evidence_units_removed_count": len(removed_units_info),
            "total_tokens_consumed": total_tokens,
            "all_units": all_units_info,
            "kept_units": kept_units_info,
            "removed_units": removed_units_info
        }