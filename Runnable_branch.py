from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda , RunnableBranch

model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"

# LLM setup
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)


parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt , model , parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500 , RunnableSequence(prompt2 , model , parser)),
    RunnablePassthrough
)


final_chain = RunnableSequence(report_gen_chain , branch_chain)

print(final_chain.invoke({'topic' : 'Russia vs Ukarine'}))