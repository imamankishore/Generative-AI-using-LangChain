from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"

# LLM setup
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)
model = ChatHuggingFace(llm=llm)

# Load document (ensure docs.txt exists!)
loader = TextLoader(r"C:\Users\itsam\LangChain Models\Runnables\docs.txt")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embeddings + Vector store
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(docs, embeddings)

# Retriever
retriever = vector_store.as_retriever()

# Manual retrieval
query = "What are key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Pass to LLM
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = model.predict(prompt)

print("Answer:", answer)
