import os
from contextlib import asynccontextmanager 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.rag_pipeline import QueryAwareRAG
from src.data.demo_loader import load_demo_datasets

# Load env variables
load_dotenv()
token = os.getenv("HF_TOKEN")

# Global variables to hold state
pipeline = None
demo_data = None
query_to_dataset = {}
current_indexed_key = None

@asynccontextmanager
async def lifespan(app: FastAPI):
  """
  Handles startup and shutdown events for the FastAPI application.
  """
  global pipeline, demo_data, query_to_dataset
  print("Booting up heavy ML models... (This will take a moment)")
  
  pipeline = QueryAwareRAG(token=token)
  demo_data = load_demo_datasets()
  
  # Build a fast O(1) lookup map for queries -> dataset
  for ds_id, ds_info in demo_data.items():
    for q in ds_info["queries"]:
      query_to_dataset[q] = ds_id
          
  print("✓ FastAPI Server Ready.")
    
  yield # Yield control back to FastAPI to start accepting requests

# Initialize App with the new lifespan context
app = FastAPI(
  title="Query-Aware RAG Compression API",
  lifespan=lifespan
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class LoadDatasetRequest(BaseModel):
  dataset_id: str

class QueryRequest(BaseModel):
  query: str
  top_k: int = 4
  compare_original: bool = True
  use_coarse: bool = True
  use_fine: bool = True


@app.get("/datasets")
def get_datasets():
  if not demo_data:
    raise HTTPException(status_code=500, detail="Demo data not loaded.")
  
  return {
    dataset_id: {
      "name": info["name"],
      "queries": info["queries"]
    }
    for dataset_id, info in demo_data.items()
  }

@app.post("/dataset/load")
def load_dataset(request: LoadDatasetRequest):
  """Legacy/Fallback manual loading. Mostly ignored by auto-indexing UI."""
  global current_indexed_key
  
  if request.dataset_id not in demo_data:
    raise HTTPException(status_code=404, detail="Dataset not found")
      
  docs = demo_data[request.dataset_id].get("global_documents", [])
  if not docs:
    return {"status": "skipped", "message": "This dataset uses query-specific auto-indexing."}
      
  index_key = f"global_{request.dataset_id}"
  if current_indexed_key == index_key:
    return {"status": "success", "message": "Already loaded."}
      
  pipeline.retriever.index_documents(docs)
  current_indexed_key = index_key
  
  return {"status": "success", "message": f"Indexed {len(docs)} documents."}

@app.post("/query")
def run_query(request: QueryRequest):
    """Executes the pipeline with Auto-Indexing."""
    global current_indexed_key
    
    # 1. Figure out which dataset this query belongs to
    dataset_id = query_to_dataset.get(request.query)
    
    if not dataset_id:
        # FIX: Fallback to the first available loaded dataset if user typed a custom query
        dataset_id = list(demo_data.keys())[0] if demo_data else None
        
    if not dataset_id:
        raise HTTPException(status_code=500, detail="No datasets available in backend.")
        
    ds_info = demo_data[dataset_id]
    
    # 2. Determine which documents need to be in the index
    if request.query in ds_info.get("query_documents", {}):
        docs_to_index = ds_info["query_documents"][request.query]
        index_key = f"query_{request.query}"
    else:
        docs_to_index = ds_info.get("global_documents", [])
        index_key = f"global_{dataset_id}"
        
    # 3. AUTO-INDEXING: If the retriever doesn't have these docs, index them now
    if current_indexed_key != index_key:
        print(f"\n[Auto-Index] Swapping index for key: {index_key}...")
        pipeline.retriever.index_documents(docs_to_index)
        current_indexed_key = index_key

    # 4. Run the Pipeline
    try:
        result = pipeline.run(
            query=request.query,
            top_k=request.top_k,
            compare_original=request.compare_original,
            use_coarse=request.use_coarse,
            use_fine=request.use_fine
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="127.0.0.1", port=8000)