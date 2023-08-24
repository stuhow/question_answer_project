import pickle

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory

from langchain.memory import StreamlitChatMessageHistory

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the most recent state of the union address.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"

Context: {context}
---
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])


def load_retriever():
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_basic_qa_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = load_retriever()
    model = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
    return model


def get_condense_prompt_qa_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT})
    return model
