
from fastapi import FastAPI
import uvicorn

#导入分发路由
from fastapi_router.fenfa1 import fenfa1_router

app = FastAPI()

#将路由分发到app中
app.include_router(fenfa1_router, prefix="/fenfa1", tags=["分发"])

@app.get("/")
def hello():
    return {"message": "Hello, World"}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)

