import os
import torch
from utils import *
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ----------------- 1. 文档加载与处理 (不变) -----------------
print_step(1, "加载文档...")
base_dir = "documents"
documents = []
for file_name in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file_name)
    if file_name.endswith(".txt"):
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())
    elif file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file_name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
print_success("文档加载完成")

print_step(2, "文档分块...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)
print_success("文档分块完成")


# ----------------- 2. 嵌入模型与向量数据库 (CORRECTED) -----------------
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



print_step(3, "构建并存储向量数据库...")

# 使用 .from_documents() 一步完成数据库构建和文档存储
# LangChain 会自动处理集合的创建
vectorstore = Qdrant.from_documents(
    documents=chunked_documents,      # 您已经分好块的文档
    embedding=embedding_model,        # 您已经加载的嵌入模型
    location=":memory:",              # 使用内存数据库，也可以换成 path="./qdrant_db" 进行本地持久化
    collection_name="my_documents",   # 指定集合的名称
    force_recreate=True,              # 强制重新创建集合，确保每次运行都是新的
)
print_success("构建向量数据库完成")




# ----------------- 3. 多模型初始化 (已优化) -----------------
print_step(4, "初始化大语言模型...")
load_dotenv()

available_models = {}
model_configs = {
    "qwen-turbo": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": os.getenv("DASHSCOPE_API_URL"),
        "provider": "Dashscope"
    },
    "gpt-4o-mini": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_API_URL"),
        "provider": "OpenAI"
    },
    "deepseek-chat": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_API_H"),
        "provider": "DeepSeek"
    },
    "moonshot-v1-8k": {
        "model_name": "moonshot-v1-8k",
        "api_key": os.getenv("MOONSHOOT_API_KEY"),
        "base_url": os.getenv("MOONSHOOT_API_URL"),
        "provider": "Moonshot"
    }
}

# 优化：我们不再需要 default_llm 对象，只需要默认模型的名称 (key)
default_model_key = "" 

for model_id, config in model_configs.items():
    if config["api_key"] and config["base_url"]:
        try:
            model_name = config.get("model_name", model_id)
            chat_model = ChatOpenAI(
                model=model_name,
                base_url=config["base_url"],
                api_key=SecretStr(config["api_key"]),
                temperature=0.0
            )
            available_models[model_id] = chat_model
            # 将第一个成功加载的模型key设置为默认key
            if not default_model_key:
                default_model_key = model_id 
            print_success(f"成功加载模型: {model_id} ({config['provider']})")
        except Exception as e:
            print_error(f"加载模型 {model_id} 失败: {e}")

if not available_models:
    print_error("没有任何大语言模型被成功加载，请检查 .env 文件配置。")
    exit(1)


# ----------------- 4. 构建可配置的问答链 (已修正) -----------------
print_step(5, "构建问答链...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

template = """
根据以下检索到的上下文信息，简洁地回答问题。
如果上下文中没有相关信息，就直接说你不知道。
上下文:
{context}

问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

# --- 代码修正部分 ---
# 创建一个简单的函数来根据配置选择LLM，避免configurable_alternatives的重复键问题
def get_llm_for_config(config):
    """根据配置返回相应的LLM实例"""
    llm_key = config.get("configurable", {}).get("llm", default_model_key)
    return available_models.get(llm_key, available_models[default_model_key])

# 使用自定义类创建一个可配置的LLM选择器

def configurable_llm_func(inputs):
    """可配置的LLM函数，根据运行时配置选择合适的LLM"""
    # 这个函数会在运行时被调用，此时可以访问配置
    return inputs

# 我们将在链中直接处理LLM选择，而不是使用configurable_alternatives
configurable_llm = RunnableLambda(configurable_llm_func)
# --- 修正结束 ---

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 创建一个自定义的可配置RAG链类
class ConfigurableRAGChain:
    def __init__(self, retriever, prompt, output_parser, available_models, default_model_key):
        self.retriever = retriever
        self.prompt = prompt
        self.output_parser = output_parser
        self.available_models = available_models
        self.default_model_key = default_model_key

    def with_config(self, configurable):
        """返回一个配置了特定LLM的链实例"""
        return ConfiguredRAGChain(
            self.retriever,
            self.prompt,
            self.output_parser,
            self.available_models,
            configurable,
            self.default_model_key
        )

    def invoke(self, question):
        """使用默认配置调用链"""
        return self.with_config({"llm": self.default_model_key}).invoke(question)

class ConfiguredRAGChain:
    def __init__(self, retriever, prompt, output_parser, available_models, config, default_model_key):
        self.retriever = retriever
        self.prompt = prompt
        self.output_parser = output_parser
        self.available_models = available_models
        self.config = config
        self.default_model_key = default_model_key

    def invoke(self, question):
        """执行RAG链"""
        # 1. 检索相关文档
        docs = self.retriever.invoke(question)
        context = format_docs(docs)

        # 2. 构建提示
        prompt_value = self.prompt.invoke({"context": context, "question": question})

        # 3. 选择并调用LLM
        llm_key = self.config.get("llm", self.default_model_key)
        llm = self.available_models.get(llm_key, self.available_models[self.default_model_key])
        response = llm.invoke(prompt_value)

        # 4. 解析输出
        return self.output_parser.invoke(response)

# 创建RAG链实例
rag_chain = ConfigurableRAGChain(retriever, prompt, output_parser, available_models, default_model_key)
print_success("可配置问答链构建完成")


# ----------------- 6. FastAPI 应用 (完整版) -----------------
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# --- Pydantic 模型定义 ---
# Pydantic 模型用于定义请求和响应的数据结构，FastAPI会用它们来做数据验证和序列化

class QuestionRequest(BaseModel):
    """定义发往 /ask 接口的请求体结构"""
    question: str  # 必须包含一个名为 'question' 的字符串字段
    model: str     # 必须包含一个名为 'model' 的字符串字段，用于指定LLM

class AnswerResponse(BaseModel):
    """定义从 /ask 接口返回的响应体结构"""
    answer: str
    source_documents: List[Dict[str, Any]] # 源文档是一个列表，列表中每个元素是字典
    question: str
    model_used: str # 告知前端实际使用了哪个模型

# --- FastAPI 应用实例化 ---
app = FastAPI(
    title="智能文档问答系统", 
    description="基于AI的文档检索与问答API",
    version="1.0.0"
)

# 设置模板目录，用于加载 index.html
templates = Jinja2Templates(directory="templates")


# --- API 接口 (Endpoints) 定义 ---

@app.get("/", response_class=HTMLResponse, tags=["前端页面"])
async def read_root(request: Request):
    """
    根路径，返回前端的HTML主页面。
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get_models", tags=["API接口"])
async def get_models():
    """
    获取所有已成功加载的可用大语言模型列表。
    前端通过调用此接口来填充模型选择下拉框。
    """
    return {"models": list(available_models.keys())}

@app.post("/ask", response_model=AnswerResponse, tags=["API接口"])
async def ask_question(question_request: QuestionRequest):
    """
    处理用户提问的核心接口。
    - 接收一个包含问题和模型名称的JSON对象。
    - 使用RAG链进行检索和问答。
    - 返回一个包含答案和源文档的JSON对象。
    """
    try:
        question = question_request.question.strip()
        model_key = question_request.model
        
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        # 检查前端传来的模型key是否可用，如果不可用则使用默认key
        if model_key not in available_models:
            print_warning(f"模型 {model_key} 不可用，将使用默认模型 {default_model_key}。")
            model_key = default_model_key

        print_info(f"收到问题: '{question}' (使用模型: {model_key})")
        
        print_loading("正在检索相关文档...")
        source_documents = retriever.invoke(question)
        print_success(f"文档检索完成，找到 {len(source_documents)} 个相关片段。")

        print_loading(f"正在使用 {model_key} 生成回答...")

        # 调用链时，通过配置字典告诉 'llm' 切换器，本次调用使用哪个 key 对应的模型
        response_text = rag_chain.with_config({"llm": model_key}).invoke(question)

        print_success("回答生成完成。")
        
        source_docs = []
        if source_documents:
            for doc in source_documents:
                source_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
        
        return AnswerResponse(
            answer=response_text,
            source_documents=source_docs,
            question=question,
            model_used=model_key
        )
        
    except Exception as e:
        error_msg = f"处理问题时出错: {str(e)}"
        print_error(error_msg)
        import traceback
        print_error(f"详细错误: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)
if __name__ == "__main__":
    print_banner("智能文档问答系统")
    print_info("系统正在初始化...")
    print_model(f"已加载的AI模型: {', '.join(available_models.keys())}")
    print_server("Web界面: http://localhost:8001")
    print_server("API文档: http://localhost:8001/docs")
    print_success("系统启动完成！")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)