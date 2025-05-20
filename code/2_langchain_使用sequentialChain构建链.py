'''
SequentialChain: 顺序链
调用本地的 ollama 大模型，构建一个顺序链，实现一个问答系统
'''
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.llms.ollama import Ollama



#链1
content = "人为什么要学习？"
prompt = ChatPromptTemplate.from_template("你好，请你解释以下问题，内容：{content}")

llm = Ollama(base_url="http://localhost:11434", model="deepseek-r1:1.5b", temperature=0.5)

chain1 = LLMChain(prompt=prompt, llm=llm, output_key="gen_text", verbose=True)

#链2
content2 = "你好，请你对下面内容进行总结，内容：{gen_text}"
prompt2 = ChatPromptTemplate.from_template(content2)

chain2 = LLMChain(prompt=prompt2, llm=llm, output_key="summary", verbose=True)

#链3
content3 = "你好，请你将下面内容翻译成英文，内容：{summary}"
prompt3 = ChatPromptTemplate.from_template(content3)

chain3 = LLMChain(prompt=prompt3, llm=llm, output_key="translate", verbose=True)

#构建顺序链
Sequential_chain = SequentialChain(chains=[chain1, chain2, chain3],input_variables=["content"], output_variables=["gen_text", "summary","translate"] , verbose=True)

res = Sequential_chain(content)
print(res)
