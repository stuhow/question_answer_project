from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle


print("Loading data...")
loader = DirectoryLoader('raw_data/', glob='*.txt', loader_cls=TextLoader)
docs = loader.load()


print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                               chunk_overlap = 0)
all_splits = text_splitter.split_documents(docs)

print("Creating vectorstore...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_splits, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
