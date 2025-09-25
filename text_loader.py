# from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
# from langchain_core.output_parsers import StrOutputParser
# from langchain.prompts import PromptTemplate
# from langchain_community.document_loaders import TextLoader,PyPDFLoader


# model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"



# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_name,     
#     task="text-generation",
#     model_kwargs={"temperature": 0.7},             # stays here
#     pipeline_kwargs={"max_new_tokens": 256}        # moved here ✅
# )

# model = ChatHuggingFace(llm=llm)

# prompt = PromptTemplate(
#     template='Write a summaey for the following poem \n {poem}',
#     input_variables=['poem']
# )

# parser = StrOutputParser()


# loader = PyPDFLoader(r"C:\Users\itsam\LangChain Models\Documents\AMAN_KISHORE_RESUME_GGU.pdf")

# docs = loader.load()

# chain = prompt | model | parser

# chain.invoke({'poem' : docs[0].page_content})







from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model path
model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"

# Initialize LLM
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs={"temperature": 0.7},
    pipeline_kwargs={"max_new_tokens": 256}
)

model = ChatHuggingFace(llm=llm)

# Prompt template
prompt = PromptTemplate(
    template='Write a summary for the following text:\n{poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

# Load PDF
loader = PyPDFLoader(r"C:\Users\itsam\LangChain Models\Documents\AMAN_KISHORE_RESUME_GGU.pdf")
docs = loader.load()

# Combine all pages into one string
text = " ".join([doc.page_content for doc in docs])

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_text(text)

# Create chain
chain = prompt | model | parser

# Summarize each chunk
summaries = []
for chunk in chunks:
    summary = chain.invoke({'poem': chunk})
    summaries.append(summary)

# Combine all summaries into a final summary
final_summary = " ".join(summaries)
print(final_summary)
