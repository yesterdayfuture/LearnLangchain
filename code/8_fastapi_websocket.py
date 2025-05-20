
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI(version="0.1.0", title="FastAPI WebSocket Example")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")


if __name__ == "__main__":
    uvicorn.run("8_fastapi_websocket:app", host="0.0.0.0", port=8000)
