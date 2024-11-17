import streamlit as st
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

for msgItem in st.session_state.chatMsgHistory:
    with st.chat_message(msgItem["role"]):
        st.markdown(msgItem["content"])

with st.sidebar:
    modelSel = st.selectbox("Choose the model",
                    ["llama3.2:latest" , "gemma2:2b"],
                 index = 0,
                 )
    ollamaGen.setModel(modelSel)
    #st.session_state.chatmodel = modelSel
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

    msgHistoryList = []
    for msgHis in st.session_state.chatMsgHistory:
        msgHistoryList.append((msgHis["role"] , msgHis["content"]))

    modelRes = None
    if memoFunction == "Full memory history":
        modelRes = ollamaGen.chatResponse(userInput , msgHistoryList)
    elif memoFunction == "Summary of History":
        modelRes = ollamaGen.chatResbySummary(userInput , st.session_state.chatMsgHistory)
    if modelRes.content:
        st.session_state.chatMsgHistory.append({"role": "ai", "content": modelRes.content})
        msgHistoryList.append(("user" , modelRes.content))
        #ollamaGen.summarizeHistory(st.session_state.chatMsgHistory)
        with st.chat_message("ai"):
            st.markdown(modelRes.content)
            st.session_state.inputToken = modelRes.usage_metadata["input_tokens"]
            tokenAmnt.text(f"Here is the total input token of the prompt:  {st.session_state.inputToken}")
            #print(modelRes.usage_metadata)
