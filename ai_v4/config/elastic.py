from elasticsearch import Elasticsearch
from ai_v4.config.settings import settings

es=Elasticsearch(settings.ELASTIC_HOST, request_timeout=15, max_retries=3, retry_on_timeout=True)
