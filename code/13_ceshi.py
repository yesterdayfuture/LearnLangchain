from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd
import io
import os
from dashscope import Generation
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.utils.function_calling import convert_to_openai_tool

# from langchain.agents import create_openai_tools_agent, AgentExecutor
#通义千问api_key
os.environ["DASHSCOPE_API_KEY"] = "sk-3e8d696b9b3a403caa30db95c801f875"


@tool
def coderunner(code: str) -> dict:
    """python代码执行器"""

    #默认数据指定 r1
    run_code=PythonREPL(_globals={"df":r1})
    res=run_code.run(code)
    if res=='':
        return run_code
    else:
        return res

parser = JsonOutputParser()


r1 = pd.read_csv("data.csv")
buf = io.StringIO()
r1.info(buf=buf, memory_usage='deep', show_counts=True)
df_info = buf.getvalue()


# 提示词模版
def get_response_t(messages):
    response = Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwen-max',
        messages=messages,
        tools=[convert_to_openai_tool(coderunner)],
        result_format='message', # 将输出设置为message形式
    )
    return response

def prompt_data(content):
    system_prompt_t=f"""已知代码中数据信息如下：
{df_info}
此数据已经赋值给全局变量df，在已知df的基础上基于pandas编写代码然后调用工具coderunner完成任务,注意统计结果的变量名必须是result_data
"""
    prompt=[{"role":"system","content":system_prompt_t},\
            {"role":"user","content":"总共有多少行记录"},\
            {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'id': 'call_123','name': 'coderunner', 'arguments': '{"code": "import pandas as pd\nresult_data=len(df)"}'}}]},
            {"role": "tool", "tool_call_id": "call_123", "content": "共有 1000 行记录"}]
    prompt.append({"role":"user","content":content})
    return prompt

p_y=prompt_data("总共有多少行记录")

res=get_response_t(p_y)
print(res)

print(res)
code=parser.parse(res.output.choices[0].message['tool_calls'][0]['function']['arguments'])["code"]
# print(code)

run_res=coderunner.invoke(code)

print(run_res)
