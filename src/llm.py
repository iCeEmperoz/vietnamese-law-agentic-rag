"""LLM integration module using Gemini and Groq API via LangChain."""
import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Provider config (default to gemini if not specified)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Module-level LLM singleton cache keyed by (provider, temperature)
# Avoids re-creating LLM client on every call (was ~0.3-0.5s overhead per call)
_llm_cache: dict = {}


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Retrieve the Chat LLM client (cached singleton per temperature).

    Supports both 'gemini' and 'groq' providers based on environment settings.
    """
    global _llm_cache
    
    # Cache key includes provider and temperature to prevent collisions
    cache_key = (LLM_PROVIDER, temperature)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing. Please configure it in your .env file.")
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model=GROQ_MODEL,
                temperature=temperature,
                groq_api_key=GROQ_API_KEY
            )
            _llm_cache[cache_key] = llm
            return llm
        except ImportError:
            raise ImportError(
                "langchain-groq library is not installed. "
                "Please run `pip install langchain-groq` to use Groq API."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatGroq: {e}")
            
    else:  # Default to gemini
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Please configure it in your .env file.")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                temperature=temperature,
                google_api_key=GEMINI_API_KEY
            )
            _llm_cache[cache_key] = llm
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatGoogleGenerativeAI: {e}")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Retrieve the Embeddings client using gemini-embedding-001."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing. Please configure it in your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

