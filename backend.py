
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from database import DB_NAME
from langgraph.prebuilt import ToolNode,tools_condition

import os
import sqlite3

from langchain_core.runnables import RunnableConfig

from tools import calculator, search_web, search_documents, list_indexed_documents
from rag import list_sources
 

model = ChatOllama(
    model='dmtx/qwen3.5-9b-abliterated:latest',
    keep_alive='10m',
    temperature=0,
    reasoning=False
)


avail_tools = [calculator, search_web, search_documents, list_indexed_documents]
 
model_with_tool = model.bind_tools(avail_tools)


SYSTEM_PROMPT = """
You are a factual AI assistant with access to external tools.

IMPORTANT RULES:

1. CURRENT INFORMATION:
   - If the question relates to the uploaded documents, follow the
     DOCUMENT CONTEXT and rule 7 first.
   - Otherwise always call search_web for current or changing facts.
   - This includes current office holders, latest news,
     recent events, prices and software versions.
   - Never answer these questions using pretrained knowledge alone.

2. EVIDENCE:
   - Treat search results as your primary evidence.
   - Use only facts supported by retrieved content.
   - Never invent names, dates, statistics, events or allegations.
   - Never combine unrelated news stories or different people.
   - Do not assume that a search result title proves its contents.

3. ALLEGATIONS:
   - Distinguish confirmed facts from reported allegations.
   - Attribute allegations to their actual reporting sources.
   - Never present allegations as established facts.
   - Do not infer guilt from an investigation or accusation.

4. SOURCE HANDLING:
   - Prefer official sources and credible news organizations.
   - Cross-check important or controversial claims.
   - Include source URLs provided by search_web.
   - Never fabricate citations or URLs.

5. UNCERTAINTY:
   - If evidence is insufficient, say that you cannot verify it.
   - If sources conflict, explain the disagreement.
   - Never fill missing information using assumptions.

6. FINAL ANSWER:
   - Be concise and factual.
   - Support important factual claims with sources.
   - Do not claim that you searched unless the tool was executed.

7. DOCUMENTS:
   - When the user asks about uploaded documents, call search_documents first.
   - Always cite the source file and page returned by the tool.
   - If no relevant passage is found, say so instead of guessing.
"""

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



def chat_node(state: ChatState, config: RunnableConfig):
    tid = (config or {}).get("configurable", {}).get("thread_id")
    docs = list_sources(tid) if tid else []
    system_prompt = SYSTEM_PROMPT
    if docs:
        names = ", ".join(os.path.basename(d) for d in docs)
        system_prompt = (
            "DOCUMENT CONTEXT:\n"
            f"The user has uploaded these documents: {names}.\n"
            "Their contents are the primary source for this conversation.\n"
            "For ANY user question, first call search_documents.\n"
            "If it returns relevant passages, answer from them and cite the file.\n"
            "Only call search_web if the documents do not contain the answer.\n\n"
        ) + system_prompt
    messages = [SystemMessage(content=system_prompt), *state["messages"]]
    response = model_with_tool.invoke(messages)
    return {"messages": [response]}



# LangGraph persistent checkpointer
conn = sqlite3.connect(
    str(DB_NAME),
    check_same_thread=False
)

chkpt = SqliteSaver(conn)


graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools',ToolNode(avail_tools))

graph.add_edge(START, 'chat_node')

graph.add_conditional_edges('chat_node',tools_condition)

graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=chkpt)
