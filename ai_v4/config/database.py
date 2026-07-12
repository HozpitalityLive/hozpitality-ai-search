import psycopg2
from psycopg2.pool import SimpleConnectionPool
from ai_v4.config.settings import settings

db_pool = SimpleConnectionPool(
    1,
    10,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    dbname=settings.DB_NAME,
    user=settings.DB_USER,
    password=settings.DB_PASSWORD
)