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
├── vector_db/              # 向量数据库缓存
└── embedding_models/        # 嵌入模型目录
```

## 环境变量

创建 `.env` 文件：

```
BAILIAN_API_URL=your_bailian_api_url
BAILIAN_API_KEY=your_bailian_api_key
```

## 向量数据库构建性能分析与优化

在构建FAISS向量数据库时，我们发现性能瓶颈主要存在于以下几个方面：

### 性能瓶颈分析

1. **文档编码问题**：不同编码格式的TXT文件在读取过程中可能导致异常，需要统一转换为UTF-8。

2. **大文本分块处理**：当前使用的`RecursiveCharacterTextSplitter`以固定大小分割文档，对于大量文档处理效率较低。

3. **嵌入模型计算开销**：嵌入模型在CPU上运行时，向量化过程非常耗时。

4. **内存限制**：大量文档同时处理会导致内存占用过高，可能引起系统卡顿。

5. **并行处理缺失**：当前实现是串行处理文档，未充分利用多核CPU。

### 优化建议

1. **文档预处理优化**：
   - 使用`convert_documents_to_utf8.py`预先将所有文档转换为UTF-8编码
   - 过滤掉不需要的文档或内容，减少处理量

2. **分块策略优化**：
   - 调整`chunk_size`和`chunk_overlap`参数
   - 建议：`chunk_size=500, chunk_overlap=50`可以在性能和检索质量间取得平衡
   - 考虑使用语义分割而非简单字符分割，提高检索相关性

3. **嵌入模型加速**：
   - 如有NVIDIA GPU，启用CUDA加速
   - 尝试使用更轻量级的嵌入模型，如BERT-mini或MiniLM
   - 使用模型量化技术减少内存使用和计算开销

4. **批量处理优化**：
   - 实现批处理机制，将文档分批嵌入
   - 建议批大小：32-64，可根据内存情况调整

5. **并行处理实现**：
   ```python
   # 示例：并行文档处理
   from concurrent.futures import ProcessPoolExecutor
   
   def embed_documents_batch(docs_batch):
       return embedding_model.embed_documents([d.page_content for d in docs_batch])
   
   # 分批处理
   batch_size = 32
   batches = [chunked_documents[i:i+batch_size] for i in range(0, len(chunked_documents), batch_size)]
   
   # 并行处理
   with ProcessPoolExecutor() as executor:
       embeddings = list(executor.map(embed_documents_batch, batches))
   ```

6. **增量更新机制**：
   - 实现增量更新机制，避免每次都重建整个向量库
   - 只处理新增或修改的文档

7. **硬件建议**：
   - 增加RAM容量至少16GB
   - 使用SSD存储向量数据库
   - 如可能，使用具有CUDA支持的GPU

通过实施上述优化措施，预计可将向量数据库构建时间减少50%-70%，同时提高系统响应速度和稳定性。
