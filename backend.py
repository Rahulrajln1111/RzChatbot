
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from database import DB_NAME
import sqlite3


model = ChatOllama(
    model='dmtx/qwen3.5-9b-abliterated:latest',
    keep_alive='10m',
    temperature=0,
    reasoning=False
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    response = model.invoke(state['messages'])
    return {"messages": [response]}


# LangGraph persistent checkpointer
conn = sqlite3.connect(
    str(DB_NAME),
    check_same_thread=False
)

chkpt = SqliteSaver(conn)


graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=chkpt)
