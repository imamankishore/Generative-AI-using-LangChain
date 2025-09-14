from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel

model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"

# LLM setup
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate a tweet about the topic {topic}',
    input_variables=['topic']
)


prompt2 = PromptTemplate(
    template='Generate a Linked post about the topic {topic}',
    input_variables=['topic']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1 , model , parser),
    'linked_in' : RunnableSequence(prompt2 , model , parser)

})

result = parallel_chain.invoke({'topic' : 'AI'})
