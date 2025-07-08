
# 0. 准备文档

# 1. Loading 文档加载器，加载到Langchain中
# pip install pypdf
# pip install docx2txt
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader,TextLoader
import torch
from utils import *

# 文档加载
base_dir ="documents"

doucuments = []

for file_name in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file_name)
    if file_name.endswith(".txt"):
        loader = TextLoader(file_path, encoding='utf-8')
        doucuments.extend(loader.load())
    elif file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        doucuments.extend(loader.load())
    elif file_name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        doucuments.extend(loader.load())

# 2. Spliting: 切分、将加载的文档进行分片预处理
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10)

chunked_documents = text_splitter.split_documents(doucuments)
# print(chunked_documents)

# 3. Storrage: VectorDB --embedding
# 3-1. embeddingmodel
# 0. package install
# pip install sentence-transformers
# pip install transformers torch
# pip install langchain-huggingface
# 1. download embeddingmodel
# git lfs install
# git -c htts.sslVerify=false clone https://hf-mirror.com/moka-ai/moka-embedding-zh-base-v1

# 2. 封装
from langchain_huggingface import HuggingFaceEmbeddings

m3e_name = "./embedding_models/moka/m3e-base"
bce_name ="./embedding_models/netease-youdao"

# 指定模型运行的设备
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

# 编码时，完成归一化
encode_kwargs = {
    'normalize_embeddings': True,  # 是否归一化嵌入向量
}

# 使用HuggingFaceEmbeddings加载模型
print_loading("正在加载嵌入模型，请稍候...")
embedding_model = HuggingFaceEmbeddings(
    model_name=m3e_name, # use m3e-base for better performance,can also use bce-base
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print_success("嵌入模型加载完成")

# 3-2. storage -> vectordb
# 向量数据库很多，但是你可以自己尝试使用FAISS, ChromaDB, Weaviate等
# 这里使用 qdrant 为例
# pip install qdrant-client

from langchain_community.vectorstores import FAISS
import pickle
import os

# 向量数据库文件路径
vector_db_path = "./vector_db"
vectorstore_file = os.path.join(vector_db_path, "faiss_index")

# 检查是否已存在向量数据库
if os.path.exists(vectorstore_file + ".pkl"):
    print_info("发现已存在的向量数据库，正在加载...")
    vectorstore = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    print_success("向量数据库加载完成")
else:
    print_database("正在构建FAISS向量数据库...")
    vectorstore = FAISS.from_documents(
        documents=chunked_documents,
        embedding=embedding_model
    )
    
    # 保存向量数据库到本地
    os.makedirs(vector_db_path, exist_ok=True)
    vectorstore.save_local(vector_db_path)
    print_success("向量数据库构建并保存完成")

# 4. Retrieval: 检索 chat_model,qachain
# chat_model 使用文本生成模型，采用余弦距离来度量文本之间的相似度

# 4-1. chat_model
# pip install langchain[all] langchain-openai
# pip install pydev
'''
    apply a gemini api key here
    we use deepseek api now to see if it works
'''

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from pydantic import SecretStr
load_dotenv()

print_network("正在连接千问模型...")
try:
    chat_model = ChatOpenAI(
        model='qwen-turbo',
        base_url=os.environ["DASHSCOPE_API_URL"],
        api_key=SecretStr(os.environ["DASHSCOPE_API_KEY"])
    )
    
    # 测试API连接
    print_loading("测试API连接...")
    test_response = chat_model.invoke("你好")
    print_success("千问模型连接完成")
    
except KeyError as e:
    print_error(f"环境变量未设置: {e}")
    print_warning("请创建 .env 文件并设置 BAILIAN_API_URL 和 BAILIAN_API_KEY")
    exit(1)
except Exception as e:
    print_error(f"API连接失败: {e}")
    print_warning("请检查API配置是否正确")
    exit(1)

# 4-2. RetrievalQA chain

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=chat_model,
)
from langchain.chains import RetrievalQA

# 实例化一个 RetrivalQA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",  # 使用 "stuff" 链类型
    retriever=retriever_from_llm,
    return_source_documents=True,  # 返回源文档
    verbose=True  # 启用详细日志
)


# 5. UI: I-<qachain>-0

# question = "请问'落日六号'停留在地球的哪里？"
# response = qa_chain({"query": question})
# print("回答:", response['result'])
# print("相关文档:")
# print(response['source_documents'])
# for doc in response['source_documents']:
#     print(doc)


# 后端

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import uvicorn
from typing import List, Dict, Any

app = FastAPI(title="文档问答系统", description="基于AI的文档检索与问答API")

# 设置模板目录
templates = Jinja2Templates(directory="templates")

# 定义请求模型
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """主页面"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(question_request: QuestionRequest):
    """处理问答请求"""
    try:
        question = question_request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        print_info(f"收到问题: {question}")
        
        # 使用 qa_chain 进行问答
        print_loading("正在检索相关文档...")
        response = qa_chain({"query": question})
        print_success("文档检索完成")
        
        # 格式化源文档
        source_docs = []
        if response.get('source_documents'):
            print_info(f"找到 {len(response['source_documents'])} 个相关文档")
            for doc in response['source_documents']:
                source_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        print_success("问答处理完成")
        return AnswerResponse(
            answer=response['result'],
            source_documents=source_docs,
            question=question
        )
        
    except KeyError as e:
        error_msg = f"环境变量配置错误: {str(e)}"
        print_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except ConnectionError as e:
        error_msg = f"网络连接错误: {str(e)}"
        print_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"处理问题时出错: {str(e)}"
        print_error(error_msg)
        print_error(f"错误类型: {type(e).__name__}")
        import traceback
        print_error(f"详细错误: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "message": "文档问答系统运行正常"}

@app.get("/test_api")
async def test_api():
    """测试API连接"""
    try:
        test_response = chat_model.invoke("测试连接")
        return {"status": "success", "message": "API连接正常", "response": test_response.content}
    except Exception as e:
        return {"status": "error", "message": f"API连接失败: {str(e)}"}

@app.get("/docs_count")
async def get_documents_count():
    """获取文档数量信息"""
    try:
        # 获取向量数据库中的文档数量
        doc_count = len(chunked_documents)
        return {
            "total_documents": len(doucuments),
            "chunked_documents": doc_count,
            "status": "ready"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print_banner("智能文档问答系统")
    print_info("系统正在初始化，请耐心等待...")
    print_info("文档加载完成")
    print_model("AI模型准备就绪")
    print_server("Web界面: http://localhost:8001")
    print_server("API文档: http://localhost:8001/docs")
    print_success("系统启动完成！")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,  # 更换为8001端口
        reload=False  # 生产环境建议设为False
    )