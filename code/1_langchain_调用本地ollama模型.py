'''
学习如何调用本地的 ollama 大模型，并构建一个单独的链
'''
from langchain_community.llms import ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain



prompt = PromptTemplate(
    input_variables=["content"],
    template="你好，请你介绍一下以下问题，内容：{content}"
)

content = "现在时间是多少"
llm = ollama.Ollama(base_url="http://localhost:11434", model="deepseek-r1:7b", temperature=0.7)

# #直接使用模型
# res = llm.invoke(content)
# print(res)

#流式输出结果
for chunk in llm.stream(content):
    print(chunk, end="", flush=True)


#使用prompt和llm构建一个链
# chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# res = chain.run({"content": content})
# print(res)