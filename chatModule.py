from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class ollamaGenerator:
    model = ""
    temperature = 0.2
    memType = ""
    systemMsg = ""
    def __init__(self , model , temperature = 0.2 , systemMsg = ""):
        self.model = model
        self.temperature = temperature
        self.systemMsg = systemMsg

    def generateLLM(self):
        llm = ChatOllama(
            model = self.model,
            temperature = self.temperature
        )
        return llm

    def chatResponse(self , userMsg , msgHistory):
        msgTemplate = [("system" , "You are a helpful assistant. Answer the questions based on the history shared along with the prompt. Answer the questions according to the context in the history")]
        if len(msgHistory) > 0:
            msgTemplate = msgTemplate + msgHistory
        msgTemplate = msgTemplate + [("user" , "{inputMsg}")]
        print(msgTemplate)
       # promptTemplate = ChatPromptTemplate.from_messages(
       #     [
       #         (
       #             "system",
       #             "You are a helpful assistant"
       #         ),
       #         (
       #             "user", "{inputMsg}"
       #         ),
       # ])
        promptTemplate = ChatPromptTemplate.from_messages(msgTemplate)
        llm = self.generateLLM()
        chain = promptTemplate | llm
        modelResponse = chain.invoke({"inputMsg": userMsg})
        return modelResponse


