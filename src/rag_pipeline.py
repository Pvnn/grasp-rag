import sys
import time
import concurrent.futures
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.compression.hybrid_compressor import HybridCompressor
from src.generation.gemma_reader import GemmaRAGReader
from src.generation.reader import RAGReader

class QueryAwareRAG:
  def __init__(self, token):
    print("Initializing Query-Aware RAG Pipeline...")
    
    self.retriever = HybridRetriever()
    self.compressor = HybridCompressor(exit_token=token)
    self.reader = RAGReader()
    
    print("✓ Pipeline initialized\n")
  
  def _run_original_baseline(self, query, docs):
    """Internal helper to track time for the original_docs generation."""
    start_time = time.time()
    context_list = []
    for doc, _ in docs:
      title = doc.get('title')
      text = doc['text']
      context_list.append(f"{title}\n{text}" if title else text)
        
    context = " ".join(context_list)
    result = self.reader.generate_answer(query, context)
    end_time = time.time()
    
    # Inject the duration into the result for tracking
    result["duration"] = end_time - start_time
    return result

  def run(self, query, top_k=5, compare_original=False, use_coarse=True, use_fine=True):
    """Run end-to-end RAG pipeline."""
    print(f"\n======================================")
    print(f"Query: {query}")
    print(f"Config: Coarse={use_coarse}, Fine={use_fine}")
    print(f"======================================\n")
    
    # Step 1: Retrieval
    print(f"[1/3] Retrieving top-{top_k} documents...")
    start_retrieval = time.time()
    retrieved_docs = self.retriever.retrieve(query, top_k)
    retrieval_time = time.time() - start_retrieval
    print(f"  ✓ Retrieved {len(retrieved_docs)} docs in {retrieval_time:.2f}s\n")
    
    # --- PARALLEL ORIGINAL_DOCS EXECUTION ---
    original_future = None
    if compare_original:
      print("  [Parallel] Initiating uncompressed 'original_docs' generation in background...")
      executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
      original_future = executor.submit(self._run_original_baseline, query, retrieved_docs)

    # Step 2: Compression
    print(f"[2/3] Compressing documents...")
    start_comp = time.time()
    compressed_docs = []
    all_ep_exit_details = []
    all_quitox_details = []
    
    original_char_len = 0
    compressed_char_len = 0
    total_quitox_tokens = 0
    total_exit_tokens = 0
    total_compression_tokens = 0
    
    for i, (doc, score) in enumerate(retrieved_docs):
      title = doc.get('title')
      original_text = f"{title}\n{doc['text']}" if title else doc['text']
      result = self.compressor.compress(
        query=query, 
        context=original_text,
        #coarse_ratio=0.7,
        fine_threshold=0.5,
        use_coarse=use_coarse,
        use_fine=use_fine
      )
      
      compressed_text = result['final_text']
      compressed_docs.append(compressed_text)
      
      # Aggregate Details
      all_ep_exit_details.append({**result.get('ep_exit_details', {}), 'doc_index': i + 1})
      all_quitox_details.append({'details': result.get('quitox_details', {}), 'doc_index': i + 1})
      
      total_quitox_tokens += result['metrics']['tokens_quitox']
      total_exit_tokens += result['metrics']['tokens_exit']
      total_compression_tokens += result['metrics']['tokens_total_compression']
      original_char_len += len(original_text)
      compressed_char_len += len(compressed_text)
      
      # Informative logging for stages
      m = result['metrics']
      print(f"  Doc {i+1}: {m['original_sentence_count']} sents -> {m['coarse_sentence_count']} (Coarse) -> {m['final_sentence_count']} (Fine)")
    
    context = " ".join(compressed_docs)
    compression_time = time.time() - start_comp
    
    payload_ratio = (1 - compressed_char_len / original_char_len) * 100 if original_char_len > 0 else 0.0
    print(f"  ✓ Compressed in {compression_time:.2f}s (Ratio: {payload_ratio:.1f}%)\n")
    
    # Step 3: Generation (Compressed)
    print(f"[3/3] Generating answer (Compressed Context)...")
    start_gen = time.time()
    # We use strict_mode=False for the demo to give natural, conversational answers
    reader_result = self.reader.generate_answer(query, context, strict_mode=False)
    generation_time = time.time() - start_gen
    
    total_pipeline_time = retrieval_time + compression_time + generation_time
    
    # --- RESOLVE PARALLEL ORIGINAL_DOCS ---
    original_answer = None
    original_total_time = 0.0
    token_savings = 0.0
    time_savings = 0.0
    
    if compare_original:
      print(f"[Metrics] Finalizing 'original_docs' comparison...")
      orig_res = original_future.result()
      executor.shutdown()
      
      original_answer = orig_res["answer"]
      original_total_time = retrieval_time + orig_res["duration"]
      
      if orig_res["usage"]["prompt_tokens"] > 0:
        token_savings = (1 - reader_result["usage"]["prompt_tokens"] / orig_res["usage"]["prompt_tokens"]) * 100
      
      time_savings = original_total_time - total_pipeline_time
      
      print(f"  ✓ Original Path Time: {original_total_time:.2f}s")
      print(f"  ✓ Compressed Path Time: {total_pipeline_time:.2f}s")
      print(f"  ✓ Net Time Saved: {time_savings:.2f}s ({'FASTER' if time_savings > 0 else 'SLOWER'})\n")

    print(f"Answer: {reader_result['answer']}")
    
    return {
      'query': query,
      'answer': reader_result['answer'],
      'original_docs_answer': original_answer,
      'retrieved_docs': [{"doc_index": i+1, "text": doc['text'], "score": float(score)} for i, (doc, score) in enumerate(retrieved_docs)],
      'metrics': {
        'times': {
          'retrieval': retrieval_time,
          'compression': compression_time,
          'generation': generation_time,
          'total_pipeline': total_pipeline_time,
          'original_total': original_total_time,
          'net_time_saved': time_savings
        },
        'compression': {
          'ratio_chars': payload_ratio,
          'ratio_tokens': token_savings,
          'hf_tokens_cost': total_compression_tokens
        },
        'usage': {
          'compressed_api_tokens': reader_result['usage']['total_tokens'],
          'original_api_tokens': orig_res['usage']['total_tokens'] if compare_original else 0
        }
      },
      'ep_exit_details': all_ep_exit_details,
      'quitox_details': all_quitox_details
    }