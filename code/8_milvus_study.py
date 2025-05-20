
#使用 milvus 向量数据库
import pymilvus
from pymilvus import MilvusClient
from pymilvus import model

#要创建本地的 Milvus 向量数据库，只需实例化一个MilvusClient ，指定一个存储所有数据的文件名，如 "milvus_demo.db"
# client = MilvusClient("milvus_demo.db")

#要连接到远程 Milvus 服务器，只需实例化一个 MilvusClient ，指定服务器地址和端口，如 "localhost:19530"
client = MilvusClient(uri="http://127.0.0.1:19530",db_name="default")


client.create_database("milvus_demo")
# client.use_database("milvus_demo")

'''
在 Milvus 中，我们需要一个 Collections 来存储向量及其相关元数据。
你可以把它想象成传统 SQL 数据库中的表格。
创建 Collections 时，可以定义 Schema 和索引参数来配置向量规格，如维度、索引类型和远距离度量。
'''

if not client.has_collection(collection_name="demo_collection"):
    # client.drop_collection(collection_name="demo_collection")
    client.create_collection(
        collection_name="demo_collection",
        dimension=768,  # 向量维度
    )


# 下载一个小的预训练的嵌入模型，"paraphrase-albert-small-v2" (~50MB)。.
embedding_fn = model.DefaultEmbeddingFunction()

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = embedding_fn.encode_documents(docs)
# The output vector has 768 dimensions, matching the collection that we just created.
print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

# 每一个实体都有一个id，向量，文本和主题。
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Vector dim:", len(data[0]["vector"]))


'''
向集合中 插入 数据
'''
res = client.insert(collection_name="demo_collection", data=data)

print(res)

'''
语义搜索
向量搜索
Milvus 可同时接受一个或多个向量搜索请求。
query_vectors 变量的值是一个向量列表，其中每个向量都是一个浮点数数组

输出结果是一个结果列表，每个结果映射到一个向量搜索查询。
每个查询都包含一个结果列表，其中每个结果都包含实体主键、到查询向量的距离以及指定output_fields 的实体详细信息。
'''

query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)


'''
带元数据过滤的向量搜索

在考虑元数据值（在 Milvus 中称为 "标量 "字段，因为标量指的是非向量数据）的同时进行向量搜索。
这可以通过指定特定条件的过滤表达式来实现。让我们在下面的示例中看看如何使用subject 字段进行搜索和筛选
'''

# docs = [
#     "Machine learning has been used for drug design.",
#     "Computational synthesis with AI algorithms predicts molecular properties.",
#     "DDR1 is involved in cancers and fibrosis.",
# ]
# vectors = embedding_fn.encode_documents(docs)
# data = [
#     {"id": 3 + i, "vector": vectors[i], "text": docs[i], "subject": "biology"}
#     for i in range(len(vectors))
# ]

# client.insert(collection_name="demo_collection", data=data)

# res = client.search(
#     collection_name="demo_collection",
#     data=embedding_fn.encode_queries(["tell me AI related information"]),
#     filter="subject == 'biology'", #只查询subject为biology的数据
#     limit=2,
#     output_fields=["text", "subject"],
# )

# print(res)


'''
查询

查询()是一种操作符，用于检索与某个条件（如过滤表达式或与某些 id 匹配）相匹配的所有实体
'''
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
print(res)

'''
删除实体
如果想清除数据，可以删除指定主键的实体，或删除与特定过滤表达式匹配的所有实体。
'''

# res = client.delete(collection_name="demo_collection", ids=[0, 2])

# print(res)

# res = client.delete(
#     collection_name="demo_collection",
#     filter="subject == 'biology'",
# )

# print(res)



'''
删除 Collections
如果想删除 Collections 中的所有数据，可以通过以下方法删除 Collections
'''

# client.drop_collection(collection_name="demo_collection")





