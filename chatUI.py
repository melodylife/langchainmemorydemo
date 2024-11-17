import streamlit as st
from chatModule import ollamaGenerator as chatbot
from langchain_ollama import ChatOllama

ollamaGen = chatbot("gemma2:2b" , 0.5)
chatMsgHistory = []
if "chatMsgHistory" not in st.session_state:
    st.session_state.chatMsgHistory = []

for msgItem in st.session_state.chatMsgHistory:
    with st.chat_message(msgItem["role"]):
        st.markdown(msgItem["content"])

with st.sidebar:
    uploadFile = st.file_uploader(
        "Upload the PDF file for RAG trial"
    )
    if uploadFile is not None:
        print(f"Filename is : {uploadFile.name}")

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

    #modelRes = ollamaGen.chatResponse(userInput , msgHistoryList)
    modelRes = ollamaGen.chatResbySummary(userInput , st.session_state.chatMsgHistory)
    if modelRes.content:
        st.session_state.chatMsgHistory.append({"role": "ai", "content": modelRes.content})
        #print(modelRes.content)
        msgHistoryList.append(("user" , modelRes.content))
        #ollamaGen.summarizeHistory(st.session_state.chatMsgHistory)
        with st.chat_message("ai"):
            st.markdown(modelRes.content)
