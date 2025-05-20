
from langchain.tools import StructuredTool  # 新增关键导入

from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pydantic import BaseModel, Field


# --------------------- 步骤1：实现具体功能函数 ---------------------
def get_weather(city: str) -> str:
    """模拟天气查询API（静态数据示例）"""
    weather_data = {
        "北京": {"temperature": 22, "condition": "晴"},
        "上海": {"temperature": 25, "condition": "多云"}
    }
    return weather_data.get(city, "未找到该城市天气信息")

def calculate(operator: str, num1: float, num2: float) -> float:
    """执行基础数学运算"""
    if operator == 'add':
        return num1 + num2
    elif operator == 'subtract':
        return num1 - num2
    elif operator == 'multiply':
        return num1 * num2
    elif operator == 'divide':
        return num1 / num2 if num2 !=0 else "除数不能为零"
    else:
        return "不支持的操作符"

# --------------------- 步骤2：初始化模型并绑定工具 ---------------------
# 初始化本地Qwen模型（需提前运行ollama pull qwen:7b）
llm = OllamaFunctions(
    model="deepseek-r1:1.5b",  # 使用7B量化版本
    base_url="http://localhost:11434",
    temperature=0.7,
    format="json"
)

# weather_tool = {
#     "name": "get_weather",
#     "description": "查询城市天气",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "city": {"type": "string", "description": "需要查询的城市名称，必须是单一城市名，例如北京、上海等", "examples": ["北京", "上海"]}
#         },
#         "required": ["city"]
#         },
# }

weather_tool = {
    "name": "get_weather",
    "description": "查询城市天气",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "需要查询的城市名称，必须是单一城市名", "examples": ["北京", "上海"]}
        },
        "required": ["city"]
        },
}

math_tool = {
    "name": "calculate",
    "description": "执行基础数学运算",
    "parameters": {
        "type": "object",
        "properties": {
            "operator": {"type": "string", "description": "运算符（add, subtract, multiply, divide）"},
            "num1": {"type": "integer", "description": "第一个数字"},
            "num2": {"type": "integer", "description": "第二个数字"}
        },
        "required": ["operator", "num1", "num2"]
    },
}

# 绑定自定义工具到模型
tools = [weather_tool, math_tool]
llm_with_tools = llm.bind_tools(tools)

# --------------------- 步骤4：处理模型响应并执行函数 ---------------------
import json
def execute_function_call(query: str):
    print("模型输入：", query)

    # 发送请求到本地模型
    response = llm_with_tools.invoke(query)
    # print("模型响应：", response)
    
    # 解析工具调用结果
    if not response.tool_calls:
        return "未触发工具调用", None
    
    print("工具调用结果：", response.tool_calls)

    # 获取工具调用信息
    tool_call = response.tool_calls[0]
    func_name = tool_call["name"]
    args = tool_call["args"]  # 已自动解析为字典
    
    # 路由到对应函数
    if func_name == "get_weather":
        result = get_weather(args.get("city",None))
    elif func_name == "calculate":
        result = calculate(args["operator"], args["num1"], args["num2"])
    else:
        result = "未知工具"
    
    return func_name, result

# --------------------- 步骤5：测试运行 ---------------------
if __name__ == "__main__":

    # 测试用例1：天气查询
    query1 = "成都今天气温多少度？"
    # prompt1 = [{"role":"user","content":query1}]
    func, result = execute_function_call(query1)
    print(f"调用工具: {func}\n结果: {result}")

    # 测试用例2：数学运算
    query2 = "计算3乘以4等于多少"
    # prompt2 = [{"role":"user","content":query2}]
    func, result = execute_function_call(query2)
    print(f"调用工具: {func}\n结果: {result}")