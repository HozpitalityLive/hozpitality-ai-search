from elasticsearch import Elasticsearch
from ai_v4.config.settings import settings

es = Elasticsearch(
    settings.ELASTIC_HOST
)