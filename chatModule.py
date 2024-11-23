from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_verbose


from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list , add_messages]

class ollamaGenerator:
    model = ""
    temperature = 0.2
    memType = ""
    systemMsg = ""
    histSummary = ""
    graph = None
    memory = None

    def __init__(self , model = None , temperature = 0.2 , systemMsg = ""):
        self.model = model
        self.temperature = temperature
        self.systemMsg = systemMsg
        self.memory = MemorySaver()

    def compileGraph(self):
        memStore  = MemorySaver()
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot" , self.chatbot)
        graph_builder.add_edge(START , "chatbot")
        graph_builder.add_edge("chatbot" , END)
        self.graph = graph_builder.compile(checkpointer = memStore)

    def getGraph(self):
        return self.graph

    def chatbot(self , state: State):
        llm = self.generateLLM("")
        return {"messages": [llm.invoke(state["messages"])]}

    def generateLLM(self , model):
        llm = ChatOllama(
            model = self.model,
            temperature = self.temperature,
        )
        return llm

    def setModel(self , modelSel):
        self.model = modelSel

    def llmResponse(self , userMsg , chatHistory , histAppr):
        llmRes = ""
        if histAppr == "Full memory history":
            llmRes = self.chatResponse(userMsg , chatHistory)
        elif histAppr == "Summary of History":
            llmRes = self.chatResbySummary(userMsg , chatHistory)
        return llmRes

    def summarizeHistory(self , chatHistory):
        msghistory = [("system" , "You are helpful assistant having expertise on summarizing chat history without missing any details.") , ("user" , "Following Chat History is a history of humane and AI assistant chat. Summarize the chat history into a single summary message. Include as many specific details as you can. Respond user only with the content of the summary but no other extra content generated.  Place the summarized content after keyword summary in the format of SUMMARY: . \n\n Chat History: {chathist}")]
        llm = self.generateLLM(model = "llama3.2:latest")
        histStr = ""
        for msgItem in chatHistory:
            histStr = histStr + "role: " + msgItem["role"] + " content: " + msgItem["content"] + "\n"
        #print(histStr)
        promptTemplate = ChatPromptTemplate.from_messages(msghistory)
        chain = promptTemplate | llm
        summaryResponse = chain.invoke({"chathist" , histStr})
        print(summaryResponse.content)
        return summaryResponse.content

    def chatResbySummary(self , userMsg , msgHistory):
        chatSummary = self.summarizeHistory(msgHistory)
        msgTemplate = ["system", "You are a helpful chatbot and answer questions based on the chat history provided in the context"]
        msgTemplate = msgTemplate + [("user" , "You are chatbot designed to answer user's questions:{questionMsg}\n\n Genrate your response exclusively from the provided chat history summary: {histSummary}. \n\nIf there's no helpful details in the summary, just say you don't know or ask the user to provide the context.")]
        promptTemplate = ChatPromptTemplate.from_messages(msgTemplate)
        llm = self.generateLLM(model = "gemma2:2b")
        chain = promptTemplate | llm
        #print("This is the summary "+ chatSummary)
        #print(promptTemplate.messages)
        set_verbose(True)
        #return chain.stream({"histSummary": chatSummary , "questionMsg": userMsg})
        response =  chain.invoke({"histSummary": chatSummary , "questionMsg": userMsg})
        return response.content

    def chatResponse(self , userMsg , msgHistory):
        chatHistory = []
        for msgHisItem in msgHistory:
            chatHistory.append((msgHisItem["role"], msgHisItem["content"]))
        msgTemplate = [("system" , "You are a helpful assistant. Answer the questions based on the history shared along with the prompt. Answer the questions according to the context in the history")]
        if len(chatHistory) > 0:
            msgTemplate = msgTemplate + chatHistory
        msgTemplate = msgTemplate + [("user" , "{inputMsg}")]
        #print(msgTemplate)
        promptTemplate = ChatPromptTemplate.from_messages(msgTemplate)
        llm = self.generateLLM(model = "gemma2:2b")
        chain = promptTemplate | llm
        #return chain.stream({"inputMsg": userMsg})
        response = chain.invoke({"inputMsg": userMsg})
        return response.content


