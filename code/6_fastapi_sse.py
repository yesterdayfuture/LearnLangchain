# FastAPI 实现示例
from fastapi import FastAPI
# from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def async_generator():
    count = 0
    while True:
        # 先发送数据，再等待
        yield f"当前计数: {count}\n"
        count += 1
        await asyncio.sleep(1)

@app.get('/stream')
async def stream():
    return StreamingResponse(async_generator(), media_type="text/event-stream")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("6_fastapi_sse:app", host="127.0.0.1", port=8080)

#在终端使用下方命令进行监测 curl -N http://127.0.0.1:8080/stream