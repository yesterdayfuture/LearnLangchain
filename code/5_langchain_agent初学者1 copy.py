from langchain_community.llms.ollama import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.agents import Tool
import warnings
warnings.filterwarnings("ignore")

# 1. 初始化模型（温度设为0）
llm = Ollama(
    base_url="http://localhost:11434",
    model="deepseek-r1:7b",
    temperature=0  # 确保输出稳定性
)

# 2. 定义工具（示例中假设有天气查询和时间查询工具）
def weather_function(city_name):
    return f"{city_name} 的天气晴，温度25°C。"

def time_function(city_name):
    return f"{city_name} 现在是北京时间 15:30。"

tools = [
    Tool(
        name="WeatherTool",
        func=weather_function,
        description="一个可以查询天气的工具"
    ),
    Tool(
        name="TimeTool",    
        func=time_function,
        description="一个可以查询时间的工具"
    )
]

# 3. 重构提示模板
prompt_template = """# ReAct 结构化指令
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

**完成**：
如果已经执行完所有步骤，请在思考后输出结果。

如果问题涉及多个问题，请逐个处理，首先处理第一个问题，处理完毕后再处理第二个问题。

"""
prompt = PromptTemplate.from_template(prompt_template)

# 4. 创建带解析器的Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# 添加记忆模块
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

# 5. 执行查询，逐个处理每个问题
def execute_queries(queries):
    for query in queries:
        print(f"处理问题：{query}")
        result = agent_executor.invoke({"input": query})
        
        # 检查是否完成
        if "完成" in result["output"]:
            print("任务已完成")
        else:
            print("继续执行")

if __name__ == "__main__":
    queries = [
        "杭州的天气怎么样",
        "上海现在几点了？"
    ]
    
    execute_queries(queries)
