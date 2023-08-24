import pickle

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

template = """Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"

Context: {context}
---
Question: {question}
Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

def load_retriever():
    print("Loading retriever...")
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_basic_qa_chain():
    print("Running chain...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = load_retriever()
    model = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
    return model

qa_chain = get_basic_qa_chain()
question = "Can i drive in Argentina?"

result = qa_chain({"query": question})
print(result)
