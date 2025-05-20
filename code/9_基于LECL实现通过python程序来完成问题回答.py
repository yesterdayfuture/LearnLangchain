
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings

local_llm = ChatOllama(
    model="llama3.2:3b"
)

local_embedding = OllamaEmbeddings(
    model="mxbai-embed-large:335m"
)

# 通过代码执行 执行 python 程序
from langchain_experimental.utilities import PythonREPL
python_repl = PythonREPL()
# python_repl.run('print("hello world")')
# python_repl.run('print(3*3)')

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


template_code="请通过写python代码来回答对应问题，此问题是：{question} \n 注意仅需要通过python代码回答，不需要文字 \n 代码形式如下: ```python\n...\n```"
t_code_prompt=PromptTemplate.from_template(template_code)

# 通过代码执行 执行 python 程序
def get_code(x):
    return x.split("```python")[1].split("```")[0]

#定义链
chain_code=(t_code_prompt | local_llm | StrOutputParser() | RunnableLambda(get_code) | PythonREPL().run )

response = chain_code.invoke("1+1*9等于几")
print(response)

