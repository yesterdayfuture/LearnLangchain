

from langchain_community.llms.ollama import Ollama

from datetime import datetime
import requests
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import warnings
from pydantic import BaseModel, Field

# 忽略警告
warnings.filterwarnings("ignore")


# 定义工具
# ==================== 步骤1：定义工具 ====================

class Location(BaseModel):
    """城市/时区名称"""
    location: str = Field(description="城市/时区名称（如'上海'）")

# @tool
def get_current_time(location):
    """获取指定城市的当前时间（示例简化版）
    Args:
        location: 城市/时区名称（如"上海"）
    Returns:
        str: 格式化时间字符串
    """
    try:
        # 模拟时区转换（实际需接入时区API）
        print(f"查询{location}时间:{datetime.now()}")
        return f"{location}当前时间：{datetime.now()}"
    except Exception as e:
        return f"时间查询失败：{str(e)}"

# @tool
def get_weather(location):
    """调用天气API查询城市天气
    Args:
        location: 城市名称（如"北京"）
    Returns:
        str: 天气信息
    """
    try:
        print("get_weather: {location}天气：晴天，气温5-10摄氏度")
        return f"{location}天气：晴天，气温5-10摄氏度"
    except Exception as e:
        return f"天气查询失败：{str(e)}"

# print(get_current_time.name)
# print(get_current_time.description)
# print(get_current_time.args)



# ==================== 步骤2：初始化模型 ====================
llm = Ollama(base_url="http://localhost:11434", model="deepseek-r1:7b", temperature=0)

# ==================== 步骤3：创建Agent ====================
# tools = [get_current_time, get_weather]

tools = [
    Tool(
        name="timeTool",
        func=get_current_time,
        description="获取当前时间",
    ),
    Tool(
        name="weatherTool",
        func=get_weather,
        description="获取城市天气",
    )
]


# 设计提示模板（指导Agent决策）
prompt_template = """
# ReAct 结构化指令
你必须严格按以下格式回答问题：

**工具列表**：
{tools}

**问题**:
{input}

**思考流程**：
{agent_scratchpad}

**必须按以下顺序生成**：
Thought: 你的推理过程
Action: 工具名称（必须是 {tool_names} 之一）
Action Input: 工具的输入参数（例如城市名称）
"""


prompt = PromptTemplate.from_template(prompt_template)

# 创建React式Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

#添加记忆模块
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 包装执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,
    verbose=True  # 显示详细执行过程
)

# from langchain.agents import initialize_agent
# from langchain.agents import AgentType

 
# agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# ==================== 步骤4：执行测试 ====================
if __name__ == "__main__":
    # 示例查询
    query = "上海现在几点了？杭州的天气怎么样？"
    print(f"用户提问：{query}")
    
    # 执行Agent
    result = agent_executor.invoke({"input": query})
    print("模型原始输出:\n", result)  # 检查是否包含 "Action:"
    # 输出结果
    print("\n===== 最终响应 =====")
    print(result["output"])
