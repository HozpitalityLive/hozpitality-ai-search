from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 5432
    DB_NAME: str = "hozpitality"
    DB_USER: str
    DB_PASSWORD: str
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    ELASTIC_HOST: str = "http://127.0.0.1:9200"
    ELASTIC_INDEX: str = "hozpitality"
    OLLAMA_URL: str = "http://127.0.0.1:11434"
    DEFAULT_MODEL: str = "llama3-hoz:latest"
    PLANNER_MODEL: str = "phi3-hoz:latest"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_SEARCH_RESULTS: int = 20
    MAX_CONTEXT_RESULTS: int = 8
    SQL_MAX_ROWS: int = 50
    SQL_TIMEOUT_MS: int = 8000
    WS_HEARTBEAT: int = 30
    MEMORY_TTL: int = 7200
    API_KEY: str = ""
    CORS_ORIGINS: str = "https://hozpitality.com,https://www.hozpitality.com,http://localhost:3000"
    USE_ELASTIC: bool = True
    USE_POSTGRES: bool = True
    USE_VECTOR: bool = True

settings=Settings()
