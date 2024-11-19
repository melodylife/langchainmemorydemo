from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[list , add_messages]

graph_builder = StateGraph(State)
llm = ChatOllama(model = "gemma2:2b")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
memory = MemorySaver()
graph_builder.add_node("chatbot" , chatbot)
graph_builder.add_edge(START , "chatbot")
graph_builder.add_edge("chatbot" , END)

graph = graph_builder.compile(checkpointer = memory)
config = {"configurable": {"thread_id": "1"}}

while True:
    msgInput = input("User: ")
    if msgInput.lower() in ["quit" , "q"]:
        print("Good bye")
        break
    for event in graph.stream(
            {"messages": [("user" , msgInput)]} , config , stream_mode = "values"
    ):
        #print(event["messages"][-1].pretty_print())
        final_result = event
    print(final_result["messages"][-1].content)
        #print(event.values())
        #for value in event.values():
            #print(value["messages"])
        #    print("Assistant: " , value["messages"][-1].content)
