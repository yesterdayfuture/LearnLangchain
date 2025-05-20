
from langchain_community.llms.ollama import Ollama
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI()

#聊天agent
class Assistant:
    def __init__(self, functions):
        # 初始化llm
        self.llm = Ollama(
            base_url="http://localhost:11434",
            model="deepseek-r1:7b",
            temperature=0  # 确保输出稳定性
        )

        # 初始化工具
        self.tools = load_tools(["human","llm-math"], llm=self.llm)

        # 添加自定义函数
        if functions and len(functions) != 0:
            for func in functions:
                self.tools.append(func)

        # 初始化prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个智能助手，请根据用户的问题给出答案。请使用中文回答。"),
                ("user", "用户的问题：{input}"),
                MessagesPlaceholder(variable_name="history"),
            ]
        )


        self.agent = initialize_agent(
            tools= self.tools,
            llm= self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            prompt=self.prompt
        )

    def ask(self, question):
        return self.agent.invoke({"input":question})

# 使用示例
functions = []

def get_weather(city_name):
    return f" 城市 {city_name} 天气 是多云，温度为6-7度."

weatherTool = Tool.from_function(
    name="weatherTool",
    func=get_weather,
    description="可以根据城市的名称查询当前城市天气的工具"
)

functions.append(weatherTool)

# assistant = Assistant(functions)

# print(assistant.ask("查询北京的天气"))


@app.post("/chat")
def getChatGPTResponse(prompt):
    assistant = Assistant(functions)
    res = assistant.ask(prompt)

    # def stream_chat(prompt):
    #     yield "Assistant: 正在思考..."
    #     for response in assistant.ask(prompt):
    #         yield response
    return res

@app.websocket("/stream-chat")
async def worker_chat(websocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            # assistant = Assistant(functions)
            # res = assistant.ask(data)
            await websocket.send_text(" 接收的数据是： {data}")
        except WebSocketDisconnect:
            print("WebSocketDisconnect")
            


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)