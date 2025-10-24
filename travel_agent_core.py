# =======================================================
#               Final Stable Version (No RunnableLambda LLM)
# =======================================================
from typing import TypedDict, Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
import os

# 导入所有必要的 LangChain 和 LangGraph 模块
from langchain_core.prompts import ChatPromptTemplate
# 移除 RunnablePassthrough，改用安全节点模式
from langchain_core.messages import AIMessage, HumanMessage 

# 核心：直接使用 DashScope SDK
import dashscope  
# ------------------------------------------------------------------

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings 
from langgraph.graph import StateGraph, END 

# --- 1. Define Agent State TypedDict ---
class AgentState(TypedDict):
    """
    Defines the state of the Agent.
    messages: Stores conversation history.
    retrieved_docs: Stores documents retrieved from Faiss.
    input_question: 用于存储 RAG 决策结果的字段。
    """
    messages: Annotated[List[AnyMessage], add_messages]
    retrieved_docs: List[str]
    input_question: str # 用于存储决策字符串 ("retrieve" 或 "generate_only")

# --- 2. Configuration and Model Initialization ---
FAISS_INDEX_PATH = "faiss_index_australia"
LLM_MODEL = "qwen-max" 
EMBEDDING_MODEL = "text-embedding-v2"

# Initialize Embedding Model
embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)

# Load Faiss Vector Store
vectorstore = None
try:
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True 
    )
    print(f"Successfully loaded Faiss vector store: {FAISS_INDEX_PATH}")
except Exception as e:
    print(f"ERROR: Could not load Faiss vector store. Check path and files. Error: {e}")

# --- 3. Helper Function: Format History ---
def format_chat_history_for_llm(messages: List[AnyMessage]) -> str:
    """Converts LangGraph message list to a string format for LLM context."""
    formatted_history = []
    for message in messages:
        if message.type == "human":
            formatted_history.append(f"Human: {message.content}")
        elif message.type == "ai":
            formatted_history.append(f"AI: {message.content}")
    return "\n".join(formatted_history)


# =======================================================
#               Core Node Definitions (Direct DashScope Calls)
# 核心逻辑在 travel_agent_core.py 中，我们使用了 LangGraph 定义了一个清晰的流程：
# RAG 决策、文档检索、RAG 生成和通用生成四个节点。所有 LLM 调用均通过 DashScope SDK 实现。
# =======================================================

# --- 3.1 Node: General Response (LLM Only) ---
def generate_response_only(state: AgentState) -> dict:
    """
    【通用生成节点】
    处理通用聊天或常识性问题。如果 RAG 决策节点判断不需要检索，则路由至此。
    直接调用 DashScope SDK 实现 LLM 通用回复。
    """
    print("\n--- NODE: generate_only (General/Chat Mode) ---")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a friendly Australian travel information assistant. Please answer the question based on the conversation history."),
            ("placeholder", "{messages}"),
        ]
    )
    
    # 1. Generate the Prompt Value and convert to List[Message]
    prompt_value = prompt.invoke({"messages": state["messages"]})
    messages_for_llm = prompt_value.to_messages() 
    
    # 2. Convert to DashScope format
    dashscope_messages = []
    for msg in messages_for_llm:
        if msg.type == 'human':
            dashscope_messages.append({'role': 'user', 'content': msg.content})
        elif msg.type == 'ai':
            dashscope_messages.append({'role': 'assistant', 'content': msg.content})

    # 3. Call DashScope SDK
    try:
        response = dashscope.Generation.call(
            model=LLM_MODEL,
            messages=dashscope_messages,
            result_format='message',
            temperature=0.1
        )
        content = response.output.choices[0].message.content
        response_message = AIMessage(content=content)
    except Exception as e:
        response_message = AIMessage(content=f"Sorry, an error occurred during DashScope API call in generate_only: {e}")

    return {"messages": [response_message]}


# --- 3.2 Node: RAG Decision Execution (Now returns a dict for safety) ---
def check_rag(state: AgentState) -> dict:
    """
    【RAG 决策节点】 (FIXED for InvalidUpdateError)
    执行 LangGraph 路由决策逻辑。它返回一个字典 (dict) 来安全地更新状态，
    将决策字符串 ("retrieve" 或 "generate_only") 存入 'input_question' 字段。
    """
    print("\n--- NODE: check_rag (Decision Execution Node) ---")
    
    latest_question = state["messages"][-1].content 
    history_str = format_chat_history_for_llm(state["messages"][:-1])

    # 1. Construct the routing instruction text
    instruction_text = (
        "You are an RAG routing system. Determine if retrieval from the dedicated knowledge base is required."
        "If the question is about specific Australian travel facts (e.g., attractions, visas, transport, culture), reply only with 'retrieve'."
        "Otherwise (e.g., casual chat, common knowledge, greetings), reply only with 'generate_only'."
        f"\n\nHistory:\n{history_str}\n\nUser Question: {latest_question}"
    )
    
    # 2. Construct DashScope message format
    dashscope_messages = [{'role': 'user', 'content': instruction_text}]
    
    # 3. Call DashScope SDK
    try:
        response = dashscope.Generation.call(
            model=LLM_MODEL,
            messages=dashscope_messages,
            result_format='message',
            temperature=0.1
        )
        
        if (response and response.output and 
            response.output.choices and 
            response.output.choices[0].message):
            
            next_node_raw = response.output.choices[0].message.content.strip().lower()
        else:
            print("Warning: DashScope routing call failed or returned empty. Defaulting to 'generate_only'.")
            next_node_raw = "generate_only"

    except Exception as e:
        print(f"Fatal Error: Exception during DashScope routing call: {e}. Defaulting to 'generate_only'.")
        next_node_raw = "generate_only"


    # 4. Parse result and return a DICT (Crucial fix for InvalidUpdateError)
    if "retrieve" in next_node_raw:
        decision = "retrieve"
    else:
        decision = "generate_only"

    print(f"Decision Stored in State: {decision}")
    return {"input_question": decision} # 关键：返回字典，用 input_question 字段传递决策


# --- 4.1 Node: RAG Retrieval ---
def retrieve(state: AgentState) -> dict:
    """
    【文档检索节点】
    如果 RAG 决策节点判断需要检索，则路由至此。
    从 Faiss 向量库中检索与最新问题最相关的文档片段。
    """
    print("\n--- NODE: retrieve (Executing RAG Retrieval) ---")
    
    latest_question = state["messages"][-1].content
    
    if vectorstore is None:
        print("ERROR: Vector store not loaded, cannot perform retrieval.")
        return {"retrieved_docs": []}

    docs = vectorstore.similarity_search(latest_question, k=3) 
    retrieved_content = [doc.page_content for doc in docs]
    
    print(f"Retrieved {len(retrieved_content)} related documents.")
    
    return {"retrieved_docs": retrieved_content}


# --- 4.2 Node: RAG Generation ---
def generate_response_rag(state: AgentState) -> dict:
    """
    【RAG 生成节点】
    接收检索到的文档作为上下文，然后调用 DashScope LLM 生成最终回复。
    强制 LLM 严格基于提供的上下文回答，用于专业知识问答。
    """
    print("\n--- NODE: generate_rag (RAG Generation Mode) ---")
    
    retrieved_docs_str = "\n---\n".join(state["retrieved_docs"])
    
    # RAG prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are a professional Australian travel consultant. Please strictly use the provided [CONTEXT] and chat history to answer the user's question."
             "If the answer is not found in the context, state honestly that you cannot find the information. Do not invent facts."
             "\n\n[CONTEXT]:\n{context}"
            ),
            ("placeholder", "{messages}"), 
        ]
    )
    
    # 1. Generate the Prompt Value and convert to List[Message]
    prompt_value = prompt.invoke({
        "context": retrieved_docs_str,
        "messages": state["messages"]
    })
    messages_for_llm = prompt_value.to_messages()

    # 2. Convert to DashScope format
    dashscope_messages = []
    for msg in messages_for_llm:
        if msg.type == 'human':
            dashscope_messages.append({'role': 'user', 'content': msg.content})
        elif msg.type == 'ai':
            dashscope_messages.append({'role': 'assistant', 'content': msg.content})

    # 3. Call DashScope SDK
    try:
        response = dashscope.Generation.call(
            model=LLM_MODEL,
            messages=dashscope_messages,
            result_format='message',
            temperature=0.1
        )
        content = response.output.choices[0].message.content
        response_message = AIMessage(content=content)
    except Exception as e:
        response_message = AIMessage(content=f"Sorry, an error occurred during DashScope API call in generate_rag: {e}")

    return {"messages": [response_message]}


# --- 4.3 Main Function: Build and Run Agent ---
def run_agent_chat():
    # 1. Create StateGraph
    workflow = StateGraph(AgentState)

    # 2. Add Nodes 
    # LangGraph 流程的四个核心节点：RAG 决策 ("check_rag")、文档检索 ("retrieve")、RAG 生成 ("generate_rag") 和通用生成 ("generate_only")。
    workflow.add_node("check_rag", check_rag) # check_rag 现在是一个执行节点，返回字典
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_rag", generate_response_rag)
    workflow.add_node("generate_only", generate_response_only)

    # 3. Set Entry Point
    workflow.set_entry_point("check_rag")

    # 4. Add Normal Edges
    workflow.add_edge("generate_rag", END)
    workflow.add_edge("generate_only", END)
    workflow.add_edge("retrieve", "generate_rag")

    # 5. Add Conditional Edges
    workflow.add_conditional_edges(
        "check_rag",  
        # 关键FIX：从状态中读取 check_rag 节点先前写入的决策字符串
        lambda state: state.get("input_question"), 
        {
            "retrieve": "retrieve",
            "generate_only": "generate_only"
        }
    )

    # 6. Compile the Graph
    app = workflow.compile()

    # 7. Run Chatbot Main Loop
    print("\n--- Australian Travel Agent Started ---")
    print("Type 'exit' to quit the chat.")
    
    while True:
        user_input = input("\nHuman: ")
        if user_input.lower() == 'exit':
            break

        initial_message = HumanMessage(content=user_input)
        
        # 运行图并打印每一步的状态变化
        for step in app.stream({"messages": [initial_message]}):
            for key, value in step.items():
                 if key != "__end__":
                    print(f"  -> Executing Node: {key}") 
        
        # 使用 invoke 获取最终回复 
        # Note: We must invoke again because the stream doesn't return the final state,
        # but the stream helps visualize the step-by-step execution.
        final_state = app.invoke({"messages": [initial_message]})
        final_response_message = final_state["messages"][-1].content
        
        print(f"\nAI: {final_response_message}")
        
if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY environment variable is not set! Please run: set DASHSCOPE_API_KEY=\"Your key\" (or $env:DASHSCOPE_API_KEY=\"Your key\" in PowerShell)")
    else:
        run_agent_chat()