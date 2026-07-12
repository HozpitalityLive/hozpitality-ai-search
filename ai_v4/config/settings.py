from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    ELASTIC_HOST: str = "http://elasticsearch:9200"
    ELASTIC_INDEX: str = "hozpitality"

    OLLAMA_URL: str = "http://ollama:11434"
    DEFAULT_MODEL: str = "llama3-hoz:latest"

    MAX_SEARCH_RESULTS: int = 20
    MAX_CONTEXT_RESULTS: int = 10

    WS_HEARTBEAT: int = 30

    class Config:
        env_file = ".env"

settings = Settings()