import requests

class RAGReader:
    def __init__(self, model_name="llama3.1:8b"):
        print(f"Initializing RAGReader with Ollama {model_name}...")
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Test if Ollama is running and model exists
        try:
            test_response = requests.post(self.ollama_url, 
                json={
                    'model': self.model_name,
                    'prompt': 'test',
                    'stream': False
                },
                timeout=60
            )
            if test_response.status_code == 200:
                print(f"Using Ollama backend with {model_name}\n")
            else:
                error_data = test_response.json()
                error_msg = error_data.get('error', 'Unknown error')
                if 'not found' in error_msg.lower():
                    raise RuntimeError(
                        f"Model '{model_name}' not found. Install it with:\n"
                        f"  ollama pull {model_name}"
                    )
                raise RuntimeError(f"Ollama error: {error_msg}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama is not running. Start Ollama locally first.")
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama took too long to respond. Is it running?")

    def generate_answer(self, query, context, max_new_tokens=100, strict_mode=True):
        """
        Generate answer given query and context.
        
        Args:
            query: Question string
            context: Context string (compressed documents)
            max_new_tokens: Maximum tokens to generate
            strict_mode: If True, forces the model to output ONLY the exact entity (for benchmarking).
            
        Returns:
            Dictionary containing the generated answer and token usage statistics.
        """
        if strict_mode:
            prompt = f"""You are a strict question-answering system.
Answer the question based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
- Output ONLY the exact answer (e.g., the specific entity, name, date, or "yes"/"no").
- Do NOT write full sentences.
- Do NOT include conversational filler.
- If the answer is multiple items, list them concisely (e.g., "Item 1 and Item 2").
- If the context does not contain the answer, output EXACTLY "Insufficient context" and nothing else.

Context: {context}

Question: {query}

Exact Answer:"""
        else:
            prompt = f"""You are a helpful assistant that answers questions ONLY using the provided context. Do not use any external knowledge.

Context: {context}

Question: {query}

Answer (using only the context above):"""
        
        try:
            response = requests.post(self.ollama_url,
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0,
                        'num_predict': max_new_tokens,
                        'top_p': 1.0
                    }
                },
                timeout=120 
            )
            
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get('error', 'Unknown error')
                raise RuntimeError(f"Ollama API error: {error_msg}")
            
            result = response.json()
            
            # Extract token usage from Ollama's response schema
            prompt_tokens = result.get('prompt_eval_count', 0)
            completion_tokens = result.get('eval_count', 0)
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                "answer": result['response'].strip(),
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            
        except requests.exceptions.Timeout:
            raise RuntimeError("Generation timed out (>120s)")
        except KeyError:
            raise RuntimeError(f"Unexpected response format: {response.text}")