from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"

# LLM setup
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)
model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parse = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}' ,
    input_variables=['text']
)


chain = RunnableSequence(prompt1,model,parse , prompt2 ,model , parse)

result = chain.invoke({'topic':'AI'})

print(result)