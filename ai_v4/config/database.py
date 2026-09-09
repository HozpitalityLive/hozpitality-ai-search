from contextlib import contextmanager
from psycopg2.pool import SimpleConnectionPool
from ai_v4.config.settings import settings

db_pool=SimpleConnectionPool(
    1,10,host=settings.DB_HOST,port=settings.DB_PORT,dbname=settings.DB_NAME,
    user=settings.DB_USER,password=settings.DB_PASSWORD
)

@contextmanager
def connection():
    conn=db_pool.getconn()
    try: yield conn
    finally: db_pool.putconn(conn)
