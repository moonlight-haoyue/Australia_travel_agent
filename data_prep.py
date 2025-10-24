import os
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

# --- 配置 (确保与 travel_agent_core.py 中的配置一致) ---
# 1. 设置文档路径 (指向我们刚刚创建的文件)
DOC_PATH = "./docs/australia_info.txt" 
# 2. 设置向量库保存路径
FAISS_INDEX_PATH = "faiss_index_australia"
# 3. Embedding 模型
EMBEDDING_MODEL = "text-embedding-v2"

def create_faiss_index():
    # 1. 加载文档
    print(f"--- 1. Loading file: {DOC_PATH} ---")
    try:
        loader = UnstructuredFileLoader(DOC_PATH)
        documents = loader.load()
    except FileNotFoundError:
        print(f"Error: File not found {DOC_PATH}，please make sure docs/australia_info.txt  file has been created。")
        return

    # 2. 切割文本 (Chunking)
    print("--- 2. Dividing the text into smaller chunks (Chunking) ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # 可以根据内容调整
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"The original document was split into {len(texts)} text chunks")

    # 3. 初始化 DashScope Embedding 模型
    if not os.getenv("DASHSCOPE_API_KEY"):
         print("Error：DASHSCOPE_API_KEY The environment variable is not set！Please run set DASHSCOPE_API_KEY=\"Your key\"")
         return
         
    print("--- 3. Initializing DashScope Embedding model ---")
    embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)

    # 4. 创建和保存 FAISS 向量库
    print(f"--- 4. Creating and saving the FAISS vector database to {FAISS_INDEX_PATH} ---")
    # 注意：首次创建会调用 DashScope Embedding API
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    print("--- Vector database created successfully！---")

    # 5. 验证是否可以加载
    print("--- 5. Verifying vector library loading ---")
    FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("--- Verification successful. Data preparation complete. ---")

if __name__ == "__main__":
    create_faiss_index()