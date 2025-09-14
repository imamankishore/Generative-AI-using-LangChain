from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda

model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"

# LLM setup
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)
model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template='Write a joke on {topic}',
    input_variables=['topic']
)


def word_count(text):
    return len(text.split())

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1 , model , parser)

parallel_chain  = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableLambda(word_count)
})

final_chain= RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic' : 'Cricket'})

print(result)