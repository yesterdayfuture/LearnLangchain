
import os
#通义千问api_key
os.environ["DASHSCOPE_API_KEY"] = "sk-3e8d696b9b3a403caa30db95c801f875"

from dashscope import Generation


'''
与langchain工具的结合使用
'''

from langchain_core.tools import tool

@tool
def multiply(first_int: int, second_int: int) -> int:
    """将两个整数相乘。"""
    return first_int * second_int * 2

@tool
def add(first_int: int, second_int: int) -> int:
    "将两个整数相加。"
    return first_int + second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "对底数求指数幂。"
    return base**exponent

tools = [multiply, add, exponentiate]

#convert_to_openai_tool: 将函数转换为OpenAI工具
#即将tool工具函数转为字典
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.output_parsers import JsonOutputParser,JsonOutputKeyToolsParser
from langchain.tools.render import render_text_description,render_text_description_and_args
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()

render_tools = [i.name for i in tools]

#定义模版
def prompt_react(query):
    PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
    p0= f"""Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take in order, should be one of {render_tools}
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {query}
        Thought:"""
    return [{"role":"system","content":PREFIX},{"role":"user","content":p0}]
 


from operator import itemgetter

#获取对应工具
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

def get_response_3t(mess):
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwen-turbo',
        messages=mess,
        tools=[convert_to_openai_tool(i) for i in tools],
        result_format='message', # 将输出设置为message形式
    )
    return response
def get_args(res_c):
    args0=res_c.split("\nAction: ")[1].split("\nAction Input:")
    return {"name":args0[0],"arguments":parser.parse(args0[1])}



p_r=prompt_react("4加5乘以6然后再算4次方然后再加35")

for i in range(0,5):
    res=get_response_3t(p_r)
    res_content=res.output.choices[0].message["content"]
    # print(res_content)
    if res_content.find("Action Input:")!=-1:
        args=get_args(res_content)
        # print(args)
        t_run=tool_chain(args)
        # print(t_run)
        # print(t_run.invoke(args))
        p_r[1]["content"]=p_r[1]["content"]+res_content+"\nObservation: "+str(t_run.invoke(args))+"\n"
    else:
        p_r[1]["content"]=p_r[1]["content"]+res_content
        break
print(p_r[1]["content"])

