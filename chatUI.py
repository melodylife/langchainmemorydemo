import streamlit as st
import time
from chatModule import ollamaGenerator as chatbot
from langchain_ollama import ChatOllama

ollamaGen = chatbot("llama3.2:latest" , 0.5)
chatMsgHistory = []

if "chatMsgHistory" not in st.session_state:
    st.session_state.chatMsgHistory = []
if "inputToken" not in st.session_state:
    st.session_state.inputToken = 0
if "chatmodel" not in st.session_state:
    st.session_state.chatmodel = ""

def streamGenerator(strContent):
    for char in list(strContent):
        yield char
        time.sleep(0.01)
        

def streamWrapper(streamLLM):
    aggregate = None
    for chunk in streamLLM:
        aggregate = chunk if aggregate is None else aggregate + chunk
        yield(chunk)
    #st.session_state.inputToken = aggregate.usage_metadata["input_tokens"]
    return aggregate

for msgItem in st.session_state.chatMsgHistory:
    with st.chat_message(msgItem["role"]):
        st.markdown(msgItem["content"])

with st.sidebar:
    modelSel = st.selectbox("Choose the model",
                    ["llama3.2:latest" , "gemma2:2b"],
                 index = 0,
                 )
    ollamaGen.setModel(modelSel)
    tokenAmnt = st.info("" , icon = "👀")
    st.divider()
    memoFunction = st.radio(
        "Choose the memory implementation approach",
        ["Full memory history", "Summary of History" , "Lang-graph implementation"],
        index = 0,
        key = "memofunction"
    )

userInput = st.chat_input("Please input message here...")

for item in chatMsgHistory:
    with st.chat_message(item["role"]):
        st.write(item["content"])

if userInput:
    st.session_state.chatMsgHistory.append({"role": "user", "content": userInput} )
    with st.chat_message("user"):
        st.write(userInput)

    modelRes = None
    streamOutput = ollamaGen.streamGenerator(userInput , st.session_state.chatMsgHistory , memoFunction)
    with st.chat_message("ai"):
        modelRes =  st.write_stream(streamWrapper(streamOutput))
        tokenAmnt.text(f"Here is the total input token of the prompt:  {st.session_state.inputToken}")
        st.session_state.chatMsgHistory.append({"role": "ai", "content": modelRes})

