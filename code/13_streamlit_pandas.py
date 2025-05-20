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

st.title("Pandas DataFrame")

#定义数据并保存
# df = pd.DataFrame({
#     "A": [1, 2, 3],
#     "B": [4, 5, 6],
#     "C": [7, 8, 9]
# })

# df.to_csv("data.csv")


# online_llm = ChatOpenAI(
#     api_key="sk-8f299497aaa74c64ad2899c85c2dcaa5",
#     model="deepseek-chat",
#     base_url="https://api.deepseek.com/v1"
# )


def get_response_t(messages):
    response = Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model='qwen-max',
        messages=messages,
        tools=[convert_to_openai_tool(coderunner)],
        result_format='message', # 将输出设置为message形式
    )
    return response

parser = JsonOutputParser()



#上传数据文件并读取
if upload_file := st.file_uploader("Upload a CSV file"):

    #获取数据
    r1 = pd.read_csv(upload_file)

    # 创建内存字符串缓冲区对象
    # 使用io.StringIO()在内存中创建临时文本缓冲区（非物理文件）
    # 优点：避免物理文件IO操作，提升性能，适用于Web应用等场景
    buf = io.StringIO()  # 初始化空的内存文本流

    # 调用DataFrame的info()方法获取数据概要信息
    # 关键参数说明：
    # buf=buf：将info输出重定向到内存缓冲区（默认输出到控制台）
    # memory_usage='deep'：精确计算内存使用（需额外计算时间）
    # show_counts=True：显示非空值计数（当数据量较大时可能影响性能）
    r1.info(buf=buf, memory_usage='deep', show_counts=True)  # 将数据信息写入缓冲区

    # 从缓冲区获取完整字符串信息
    # getvalue()方法提取缓冲区全部内容，此时：
    # - buf内容仍保留在内存中（可反复读取）
    # - 执行后缓冲区指针不会自动重置（如需复用需seek(0)）
    df_info = buf.getvalue()  # 最终得到的字符串包含：
    # 数据类型、行列数、列名、非空值统计、内存用量等结构化信息
    # st.dataframe(df)
    st.write("知识库已上传")
else:
    st.write("Please upload a CSV file.")


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



# 提示词模版
def prompt_data(content):
    try:
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
    except:
        return [{"role":"user","content":content}]


if data_input := st.chat_input("请输入你的问题："):

    p_y=prompt_data(data_input)

    st.chat_message("human").write(data_input)

    res=get_response_t(p_y)
    print(res)
    code=parser.parse(res.output.choices[0].message['tool_calls'][0]['function']['arguments'])["code"]
    # print(code)

    run_res=coderunner.invoke(code)

    result = run_res.dict()['locals']['result_data']

    st.chat_message("ai").write("运行结果如下：")

    st.write(result)


