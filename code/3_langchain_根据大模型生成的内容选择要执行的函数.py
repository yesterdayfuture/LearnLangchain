'''
OllamaFunctions是一个类，它包含了一些函数，这些函数可以用来调用ollama模型。

使用OllamaFunctions类，我们可以根据ollama模型生成的内容选择要执行的函数。

例如，我们可以使用OllamaFunctions类来调用ollama模型，然后根据ollama模型生成的内容选择要执行的函数。

functions_map是一个字典，它将ollama模型生成的内容映射到要执行的函数。

prompt是一个ChatPromptTemplate对象，它包含了一个模板，这个模板可以用来生成ollama模型的输入。
'''


from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.chains import LLMChain, SequentialChain
import json

# #加载本地模型
# llm = Ollama(base_url="http://localhost:11434", model="deepseek-r1:1.5b", temperature=0.5)

# # 定义函数1
# def function1(key):
#     print("function1")
#     res = llm.invoke(key)
#     return res.json()

# # 定义函数2
# def function2(key, value):
#     print("function2")
#     res = llm.invoke("你好，医生，我的病情详情是{key},症状是{value}")
#     return res.json()


# #函数映射字典
# functions_map = {
#     "function1": function1,
#     "function2": function2
# }

# #创建OllamaFunctions对象
# ollama_functions = OllamaFunctions(base_url="http://localhost:11434", model="deepseek-r1:1.5b")

# #绑定函数
# '''
# 使用ollama_functions.bind_tools()方法，我们可以将ollama模型生成的内容映射到要执行的函数。

# tools是一个列表，它包含了要绑定的函数。列表的每一个元素都是一个函数的定义字典，字典的键是函数的名称，值是函数的定义。
# '''
# llm_tools = ollama_functions.bind_tools(tools=[
#     { #函数1的定义
#     "name": "function1", #函数名称
#     "description": "不去帮助", #函数描述
#     "parameters": { #parameters表示函数的参数
#         "key": { #参数名称
#             "type": "string", #参数类型
#             "description": "当前行为" #参数描述
#         }
#     },
#     "required": ["key"] #required表示函数的必选参数
# },
# { #函数2的定义
#     "name": "function2",
#     "description": "前去帮助",
#     "parameters": {
#         "key": {
#             "type": "string",
#             "description": "当前行为"
#         },
#         "value": {
#             "type": "string",
#             "description": "是否留取证据"
#         }
#     },
#     "required": ["key", "value"]
# }])

# #定义prompt
# prompt = "你好，我不小心碰到了腿，特别疼，你能帮助我一下吗"

# #消息列表
# message = [
#     {"role": "system", "content": "你是一位医生，在路上行走遇到一位行人求助。"},
#     {"role": "user", "content": prompt}
# ]

# #调用OllamaFunctions对象的invoke方法
# res = llm_tools.invoke(message)

# print(res.tool_calls)

# result = "大模型未使用函数"
# while True:
#     try:

#         #根据大模型生成的内容选择要执行的函数
#         function_name = res.get("tool")
#         function_action = functions_map.get(function_name, None)
#         if function_action:
#             #执行函数
#             result = function_action(**res.get("tool_input"))
        
#     except Exception as e:
#         print(e)
#         break


'''
ai 改写
'''

# 步骤1：初始化模型（关键配置）
ollama_functions = OllamaFunctions(
    base_url="http://localhost:11434",
    model="deepseek-r1:1.5b",
    format="json"  # 必须启用JSON格式
)

# 步骤2：定义工具字典
medical_tools = [
    {
        "name": "diagnose_injury",
        "description": "初步诊断外伤情况",
        "parameters": {
            "type": "object",
            "properties": {
                "body_part": {"type": "string", "description": "受伤部位如'腿部'"},
                "pain_level": {"type": "integer", "description": "疼痛等级1-10"}
            },
            "required": ["body_part"]
        }
    },
    {
        "name": "provide_first_aid",
        "description": "提供急救建议",
        "parameters": {
            "type": "object",
            "properties": {
                "injury_type": {"type": "string", "enum": ["骨折", "擦伤", "出血"]}
            },
            "required": ["injury_type"]
        }
    }
]

# 步骤3：绑定工具到模型（网页3实现方式）
llm_tools = ollama_functions.bind_tools(tools=medical_tools)

# 步骤4：定义执行函数（实际业务逻辑）
def handle_diagnose(args):
    print(f"执行诊断：部位={args['body_part']}, 疼痛等级={args.get('pain_level', '未指定')}")
    return "建议立即冰敷并保持受伤部位静止"

def handle_first_aid(args):
    print(f"提供急救：伤情类型={args['injury_type']}")
    return "使用无菌纱布按压止血，拨打120急救电话"

# 步骤5：完整调用流程（网页4流程优化）
message = [
    {"role": "user", "content": "我的右腿剧烈疼痛无法站立"}
]
# 测试用例
test_message = [
    {"role": "user", "content": "车祸导致头部大量出血不止，昏迷不醒"}
]


try:
    # 调用模型
    response = llm_tools.invoke(message)
    
    # 解析工具调用
    #hasattr: 判断对象是否包含某个属性
    if hasattr(response, 'tool_calls'):
        print("工具调用结果:", response)
        for tool_call in response.tool_calls:
            if tool_call["name"] == "diagnose_injury":
                result = handle_diagnose(tool_call["args"])
                print("诊断结果:", result)
            elif tool_call["name"] == "provide_first_aid":
                result = handle_first_aid(tool_call["args"])
                print("急救结果:", result)
except Exception as e:
    print(f"调用失败: {str(e)}")