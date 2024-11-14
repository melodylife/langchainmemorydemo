import streamlit as st

with st.sidebar:
    uploadFile = st.file_uploader(
        "Upload the PDF file for RAG trial"
    )
    if uploadFile is not None:
        print(f"Filename is : {uploadFile.name}")

with st.chat_message("user"):
    st.write("Hello")
