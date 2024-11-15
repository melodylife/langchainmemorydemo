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

    def chatResponse(self , userMsg):
        promptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant"
                ),
                (
                    "human", "{inputMsg}"
                ),
        ])
        llm = self.generateLLM()
        chain = promptTemplate | llm
        modelResponse = chain.invoke({"inputMsg": userMsg})
        return modelResponse 


