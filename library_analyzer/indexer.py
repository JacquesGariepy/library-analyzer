import torch
import faiss
from sentence_transformers import SentenceTransformer

def load_bert_model():
    """Load the BERT model with GPU detection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

def create_faiss_index():
    """Create a FAISS index."""
    dimension = 384  # Dimension of the MiniLM embeddings
    return faiss.IndexFlatL2(dimension)
