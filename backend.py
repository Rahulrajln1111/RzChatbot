
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from database import DB_NAME
from langgraph.prebuilt import ToolNode,tools_condition

from tools import calculator,search_web

import sqlite3


model = ChatOllama(
    model='dmtx/qwen3.5-9b-abliterated:latest',
    keep_alive='10m',
    temperature=0,
    reasoning=False
)


avail_tools = [calculator,search_web]
model_with_tool = model.bind_tools(avail_tools)


SYSTEM_PROMPT = """
You are a factual AI assistant with access to external tools.

IMPORTANT RULES:

1. CURRENT INFORMATION:
   - Always call search_web for current or changing facts.
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
"""

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = [SystemMessage(content=SYSTEM_PROMPT),*state['messages']]
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
