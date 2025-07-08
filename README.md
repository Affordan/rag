# 智能文档问答系统

基于 LangChain + FAISS + 千问模型的智能文档问答系统

## 功能特性

- 📚 支持多种文档格式：PDF、DOCX、TXT
- 🤖 使用千问-turbo 大语言模型
- 💾 FAISS 向量数据库，快速检索
- 🌐 Web 界面，用户友好
- 🔄 向量数据库持久化存储

## 技术栈

- **后端**: FastAPI + LangChain
- **向量数据库**: FAISS
- **嵌入模型**: m3e-base
- **大语言模型**: 千问-turbo
- **前端**: HTML + CSS + JavaScript

## 安装依赖

```bash
pip install langchain fastapi uvicorn jinja2 python-multipart
pip install langchain-community langchain-huggingface langchain-openai
pip install faiss-cpu sentence-transformers torch
pip install python-dotenv
```

## 运行系统

```bash
python DocQA.py
```

访问: http://localhost:8000

## 项目结构

```
├── DocQA.py                 # 主程序
├── file_read_store.py       # 文档编码转换工具
├── documents/               # 文档目录
├── templates/               # 前端模板
│   └── index.html
├── vector_db/               # 向量数据库缓存
└── embedding_models/        # 嵌入模型目录
```

## 环境变量

创建 `.env` 文件：

```
BAILIAN_API_URL=your_bailian_api_url
BAILIAN_API_KEY=your_bailian_api_key
```
