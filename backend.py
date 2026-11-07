
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from database import DB_NAME
from langgraph.prebuilt import ToolNode,tools_condition

from tools import calculator

import sqlite3


model = ChatOllama(
    model='dmtx/qwen3.5-9b-abliterated:latest',
    keep_alive='10m',
    temperature=0,
    reasoning=False
)


avail_tools = [calculator]
model_with_tool = model.bind_tools(avail_tools)




class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = model_with_tool.invoke(state['messages'])
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
