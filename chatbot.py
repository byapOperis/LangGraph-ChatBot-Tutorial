import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# Load .env from the nearest location; fallback to the script directory
_dotenv_path = find_dotenv()
if not _dotenv_path:
	_dotenv_path = str(Path(__file__).resolve().parent / ".env")
load_dotenv(_dotenv_path, override=True)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

def chatbot(state: State):
    ai = llm.invoke(state["messages"])
    return {"messages": ai}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def chatbot(state: State):
    ai = llm.invoke(state["messages"])
    return {"messages": [ai]}  # return a list is safest for add_messages

def stream_graph_updates(user_input: str):
    for state in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
    ):
        last = state["messages"][-1]
        # Figure out the role for both dicts and LangChain Message objects
        role = getattr(last, "type", None) or (last.get("role") if isinstance(last, dict) else None)
        if role in ("ai", "assistant"):
            content = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else "")
            print("Assistant:", content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break


# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass