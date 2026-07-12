from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    REDIS_HOST: str
    REDIS_PORT: int

    ELASTIC_HOST: str
    ELASTIC_INDEX: str

    OLLAMA_URL: str
    DEFAULT_MODEL: str

    MAX_SEARCH_RESULTS: int = 20
    MAX_CONTEXT_RESULTS: int = 10
    WS_HEARTBEAT: int = 30

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()