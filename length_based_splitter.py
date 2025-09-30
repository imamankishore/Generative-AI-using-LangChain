from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\itsam\LangChain Models\Documents\AMAN_KISHORE_RESUME_GGU.pdf")
docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=''
)


result = splitter.split_documents(docs)

print(result[50].page_content)

