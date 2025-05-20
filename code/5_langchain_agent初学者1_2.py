

from langchain_community.llms.ollama import Ollama

import warnings
# 忽略警告
warnings.filterwarnings("ignore", category=Warning)


from langchain.agents import AgentType, initialize_agent, Tool
from langchain_community.agent_toolkits.load_tools import load_tools

# 定义工具
# ==================== 步骤1：定义工具 ====================

def sum_function(string):
    a = int(string.split()[0])
    b = int(string.split()[1])
    return a + b

sum_tool = Tool.from_function(
    name="sumTool",
    func=sum_function,
    description="计算两个数字的和,当用户需要计算两个数字的和时使用这个工具，且两个参数之间用逗号隔开，例如：1,2"
)


# ==================== 步骤2：初始化模型 ====================
llm = Ollama(base_url="http://localhost:11434", model="deepseek-r1:7b", temperature=0)

#加载工具
tools = load_tools(["human","llm-math"], llm=llm)
tools.append(sum_tool)

# tools = [sum_tool]

# ==================== 步骤3：创建Agent ====================

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

res = agent.invoke("请计算1+2的和是多少")
print(res)

# from langchain_core.output_parsers import JsonOutputParser

# output_parser = JsonOutputParser()
# output_parser.invoke(res)


