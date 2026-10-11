"""Central application configuration.

All tunable values are loaded from environment variables so the system can be
reconfigured (domain, LLM provider, embedding model, chunking, retrieval)
without touching the code. Nothing is hard-coded, and no secret has a default.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings sourced from the environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- Domain -----------------------------------------------------------
    # The whole app is domain-agnostic. Change DOMAIN_NAME to re-target it.
    domain_name: str = Field(default="Company Policy Assistant", alias="DOMAIN_NAME")

    # ---- LLM provider -----------------------------------------------------
    llm_provider: str = Field(default="nvidia", alias="LLM_PROVIDER")  # nvidia|modal|openai|anthropic|huggingface
    llm_model: str = Field(default="meta/llama-3.2-11b-vision-instruct", alias="LLM_MODEL")
    llm_fallback_model: str = Field(default="openai/gpt-oss-20b", alias="LLM_FALLBACK_MODEL")
    llm_timeout: int = Field(default=120, alias="LLM_TIMEOUT")
    llm_max_tokens: int = Field(default=2500, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    @field_validator("llm_temperature", mode="after")
    @classmethod
    def validate_llm_temperature(cls, v: float) -> float:
        if not (0.2 <= round(v, 4) <= 0.3):
            raise ValueError(f"LLM_TEMPERATURE must be between 0.2 and 0.3 (got {v}).")
        return v

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    
    # ---- Hugging Face & Modal LLM settings --------------------------------
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")
    hf_model: str = Field(default="openai/gpt-oss-20b", alias="HF_MODEL")
    hf_base_url: str = Field(default="https://router.huggingface.co/v1", alias="HF_BASE_URL")
    modal_app_name: str = Field(default="domain-specific-rag-llm", alias="MODAL_APP_NAME")
    modal_llm_url: Optional[str] = Field(default=None, alias="MODAL_LLM_URL")

    # ---- Embedding provider ----------------------------------------------
    embedding_provider: str = Field(default="nvidia", alias="EMBEDDING_PROVIDER")  # nvidia|openai|bge
    embedding_model: str = Field(default="nemotron-3-embed-1b", alias="EMBEDDING_MODEL")
    # Must match the embedding model's output dimension (see embeddings/*).
    embedding_dim: int = Field(default=2048, alias="EMBEDDING_DIM")

    # ---- NVIDIA NIM settings ---------------------------------------------
    nvidia_api_key: Optional[str] = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_model: str = Field(default="nemotron-3-embed-1b", alias="NVIDIA_MODEL")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )

    # ---- Vector database --------------------------------------------------
    vector_provider: str = Field(default="qdrant", alias="VECTOR_PROVIDER")  # qdrant|pinecone
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="rag_documents", alias="QDRANT_COLLECTION")
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="rag-documents", alias="PINECONE_INDEX")

    # ---- Chunking / retrieval --------------------------------------------
    chunk_size: int = Field(default=700, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=6, alias="TOP_K")
    # Candidates fetched before filtering/reranking (>= top_k).
    retrieval_candidates: int = Field(default=15, alias="RETRIEVAL_CANDIDATES")
    # Neighboring chunk expansion settings
    neighbor_expansion_enabled: bool = Field(default=True, alias="NEIGHBOR_EXPANSION_ENABLED")
    neighbor_expansion_window: int = Field(default=1, alias="NEIGHBOR_EXPANSION_WINDOW")
    # Minimum cosine similarity for a chunk to survive filtering.
    min_similarity: float = Field(default=0.08, alias="MIN_SIMILARITY")
    # If the best candidate scores below this, we declare "insufficient context".
    sufficiency_threshold: float = Field(default=0.12, alias="SUFFICIENCY_THRESHOLD")
    # Hard cap on characters of context sent to the LLM (keeps token use bounded).
    max_context_chars: int = Field(default=8000, alias="MAX_CONTEXT_CHARS")
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    # ---- Ingestion & OCR settings ----------------------------------------
    ocr_engine: str = Field(default="auto", alias="OCR_ENGINE")  # auto | vision | local | none
    ocr_default_dpi: int = Field(default=150, alias="OCR_DEFAULT_DPI")
    ocr_high_dpi: int = Field(default=250, alias="OCR_HIGH_DPI")
    ocr_trigger_char_threshold: int = Field(default=40, alias="OCR_TRIGGER_CHAR_THRESHOLD")

    # ---- Hybrid Search & Reranking ---------------------------------------
    hybrid_search_enabled: bool = Field(default=True, alias="HYBRID_SEARCH_ENABLED")
    rrf_k: int = Field(default=60, alias="RRF_K")
    dense_weight: float = Field(default=0.6, alias="DENSE_WEIGHT")
    sparse_weight: float = Field(default=0.4, alias="SPARSE_WEIGHT")

    # ---- Versioning -------------------------------------------------------
    prompt_version: str = Field(default="prompt_v001", alias="PROMPT_VERSION")
    dataset_version: str = Field(default="golden_dataset_v001", alias="DATASET_VERSION")

    # ---- Pricing (per 1M tokens, USD). Leave blank if unknown. ------------
    llm_input_cost_per_1m: float = Field(default=0.0, alias="LLM_INPUT_COST_PER_1M_TOKENS")
    llm_output_cost_per_1m: float = Field(default=0.0, alias="LLM_OUTPUT_COST_PER_1M_TOKENS")
    embedding_cost_per_1m: float = Field(default=0.0, alias="EMBEDDING_COST_PER_1M_TOKENS")

    # ---- Uploads ----------------------------------------------------------
    max_upload_mb: int = Field(default=25, alias="MAX_UPLOAD_MB")
    allowed_extensions: str = Field(
        default=".pdf,.txt,.md,.docx,.png,.jpg,.jpeg,.webp,.bmp,.tiff",
        alias="ALLOWED_EXTENSIONS",
    )
    documents_dir: str = Field(default="data/documents", alias="DOCUMENTS_DIR")

    @property
    def allowed_ext_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
