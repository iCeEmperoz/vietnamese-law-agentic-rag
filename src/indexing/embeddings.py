import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_embedding_model():
    """Retrieve the configured embedding model.

    Requires Google text-embedding-004 via GEMINI_API_KEY.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Please configure it in your .env file.")
        
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        print("Initializing Google GenAI Embeddings (gemini-embedding-001)...")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Google GenAI Embeddings: {e}")
