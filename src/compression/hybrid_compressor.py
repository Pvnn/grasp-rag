import torch
import time
import spacy
from typing import List, Union, Dict
from collections import defaultdict

from src.compression.quitox_filter import QuitoxCoarseFilter
from src.compression.ep_exit import EPExitCompressor
from src.eval.interfaces import SearchResult


class HybridCompressor:
    """
    The Master Module: Adaptive Two-Stage Compression Pipeline.

    Flow:
    Input Doc -> [Stage 3: QUITO-X Coarse Filter] -> Coarse Doc -> [Stage 4: EP-EXIT Fine Filter] -> Final Doc
    """

    def __init__(
        self,
        exit_token: str,
        quitox_model: str = "google/flan-t5-small",
        exit_model: str = "doubleyyh/exit-gemma-2b",
        device: str = None,
        # batch_size: int = 2,  # <--- Added batch_size for local VRAM management
    ):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("\n🔗 Initializing HYBRID COMPRESSION PIPELINE...")
        
        try:
            self.nlp = spacy.load(
                "en_core_web_sm",
                disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
            )
            self.nlp.enable_pipe("senter")
            self.nlp.max_length = 2000000
        except OSError:
            print("Downloading spacy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load(
                "en_core_web_sm",
                disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]
            )
            self.nlp.enable_pipe("senter")
            self.nlp.max_length = 2000000

        # Stage 3
        print("--- [Stage 3] Loading QUITO-X ---")
        self.quitox = QuitoxCoarseFilter(model_name=quitox_model, device=self.device)

        # Stage 4
        print("--- [Stage 4] Loading EP-EXIT ---")
        self.exit = EPExitCompressor(
            token=exit_token,
            model_name=exit_model,
            threshold=0.4,
            #batch_size=batch_size, # <--- Pass batch_size to EP-EXIT
        )

        print("✅ Hybrid Pipeline Ready.\n")

    def _split_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    def compress(
        self,
        query: str,
        context: Union[str, List[str], List[SearchResult]],
        coarse_ratio: float = 0.8,
        quitox_min_keep: int = 2,
        fine_threshold: float = 0.4,
        use_coarse: bool = True,
        use_fine: bool = True,
    ) -> Dict:

        start_time = time.time()

        if isinstance(context, str):
            sentences = self._split_sentences(context)
            # All sentences from a single anonymous doc
            dummy = SearchResult(evi_id=0, docid=0, title="", text=context)
            sentence_doc_map = [(0, dummy)] * len(sentences)
            source_docs = [dummy]

        elif (isinstance(context, list)
            and len(context) > 0
            and isinstance(context[0], SearchResult)):
            sentences = []
            sentence_doc_map = []
            source_docs = context
            for doc_idx, doc in enumerate(context):
                full_text = f"{doc.title}\n{doc.text}" if getattr(doc, 'title', None) else doc.text

                # === ADD THIS DEBUG BLOCK ===
                if doc_idx == 0:  # Only print for the first document to avoid spam
                    print("\n" + "="*50)
                    print("[DEBUG] Location 1: Pre-Split Injection")
                    print(f"Original Doc Title: {getattr(doc, 'title', 'NO TITLE FOUND')}")
                    print(f"Injected full_text starts with:\n{full_text[:150]}...")
                    print("="*50 + "\n")
                # ==============================
                doc_sents = self._split_sentences(full_text)
                sentences.extend(doc_sents)
                sentence_doc_map.extend([(doc_idx, doc)] * len(doc_sents))

        else:
            sentences = context
            dummy = SearchResult(evi_id=0, docid=0, title="", text=" ".join(context))
            sentence_doc_map = [(0, dummy)] * len(sentences)
            source_docs = [dummy]

        original_count = len(sentences)
        current_sentences = sentences
        current_doc_map = sentence_doc_map

        quitox_time = 0.0
        exit_time = 0.0
        quitox_tokens = 0
        exit_tokens = 0
        quitox_details = []

        if use_coarse:
            t1 = time.time()

            quitox_result = self.quitox.compress(
                query=query,
                sentences=current_sentences,
                compression_ratio=coarse_ratio, 
                min_keep=quitox_min_keep,
            )

            kept_indices = quitox_result["kept_indices"]   # positional, no text-matching
            current_sentences = [current_sentences[i] for i in kept_indices]
            current_doc_map   = [current_doc_map[i]   for i in kept_indices]

            quitox_tokens  = quitox_result["total_tokens_consumed"]
            quitox_details = quitox_result.get("quitox_details", [])
            quitox_time    = time.time() - t1

        stage3_count = len(current_sentences)

        if use_fine:
            parent_contexts = [
                f"{doc_obj.title}\n{doc_obj.text}" if getattr(doc_obj, 'title', None) else doc_obj.text 
                for _, doc_obj in current_doc_map
            ]

            # === ADD THIS DEBUG BLOCK ===
            if len(parent_contexts) > 0:
                print("\n" + "="*50)
                print("[DEBUG] Location 2: Parent Context to Gemma")
                print(f"First parent context sent to Gemma starts with:\n{parent_contexts[0][:150]}...")
                print("="*50 + "\n")
            # ==============================
            
            self.exit.threshold = fine_threshold

            t2 = time.time()
            exit_result = self.exit.compress_with_stats(
                query=query, 
                sentences=current_sentences, 
                parent_contexts=parent_contexts  # Passed directly
            )
            exit_time = time.time() - t2

            final_text          = exit_result["compressed_text"]
            final_sentence_count = exit_result["sentences_kept"]
            exit_tokens          = exit_result.get("total_tokens_consumed", 0)

            sent_to_doc = {s: current_doc_map[i] for i, s in enumerate(current_sentences)}

            def enrich_units(units: list) -> list:
                enriched = []
                for unit in units:
                    unit_sents = unit.get("sentences", [unit.get("text", "")])
                    doc_indices = [
                        sent_to_doc.get(s, (0, source_docs[0]))[0]
                        for s in unit_sents
                        if s in sent_to_doc
                    ]
                    doc_idx = max(set(doc_indices), key=doc_indices.count) if doc_indices else 0
                    source  = source_docs[doc_idx] if doc_idx < len(source_docs) else source_docs[0]
                    enriched.append({
                        **unit,
                        "source_doc_index": doc_idx,
                        "source_doc_title":  source.title,
                        "source_evi_id":     source.evi_id,
                    })
                return enriched

            ep_exit_details = {
                "evidence_units_total":         exit_result.get("evidence_units_total", 0),
                "evidence_units_kept_count":    exit_result.get("evidence_units_kept_count", 0),
                "evidence_units_removed_count": exit_result.get("evidence_units_removed_count", 0),
                "kept_units":    enrich_units(exit_result.get("kept_units", [])),
                "removed_units": enrich_units(exit_result.get("removed_units", [])),
            }

        else:
            final_text           = " ".join(current_sentences)
            final_sentence_count = len(current_sentences)
            ep_exit_details = {
                "evidence_units_total": 0,
                "evidence_units_kept_count": 0,
                "evidence_units_removed_count": 0,
                "kept_units": [],
                "removed_units": [],
            }

        doc_to_final_sents = defaultdict(list)

        if use_fine:
            # Walk kept_units to get final sentences with doc attribution
            for unit in ep_exit_details["kept_units"]:
                doc_idx = unit["source_doc_index"]
                for s in unit.get("sentences", []):
                    doc_to_final_sents[doc_idx].append(s)
        else:
            for i, s in enumerate(current_sentences):
                doc_idx = current_doc_map[i][0]
                doc_to_final_sents[doc_idx].append(s)

        compressed_docs = []
        for doc_idx, doc in enumerate(source_docs):
            text = " ".join(doc_to_final_sents.get(doc_idx, []))
            if text.strip():
                compressed_docs.append(SearchResult(
                    evi_id=doc.evi_id,
                    docid=doc.docid,
                    title=doc.title,
                    text=text,
                ))

        # Fallback: if grouping produced nothing but we have final_text, return it flat
        if not compressed_docs and final_text.strip():
            compressed_docs = [SearchResult(
                evi_id=0, docid=0, title="compressed", text=final_text
            )]

        total_time = time.time() - start_time
        total_compression_tokens = quitox_tokens + exit_tokens

        return {
            "final_text": final_text,
            "compressed_docs": compressed_docs,
            "metrics": {
                "original_sentence_count":  original_count,
                "coarse_sentence_count":    stage3_count,
                "final_sentence_count":     final_sentence_count,
                "time_quitox":  round(quitox_time, 2),
                "time_exit":    round(exit_time, 2),
                "time_total":   round(total_time, 2),
                "tokens_quitox":             quitox_tokens,
                "tokens_exit":               exit_tokens,
                "tokens_total_compression":  total_compression_tokens,
            },
            "quitox_details":  quitox_details,
            "ep_exit_details": ep_exit_details,
        }