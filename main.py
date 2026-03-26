from fastapi import FastAPI
from ai_server import app as app1
from ai_server_2 import app as app2

main_app = FastAPI()

main_app.mount("", app1)
main_app.mount("/v2", app2)