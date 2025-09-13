from langchain_huggingface import HuggingFacePipeline , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel ,RunnableBranch ,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from typing import  Literal

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

model_name = r"C:\Users\itsam\LangChain Models\distilbert-sst2"



llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,     
    task="text-classification",
    model_kwargs={"temperature": 0.7, "max_length": 256}
)

model = ChatHuggingFace(llm=llm)

parser =  StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback into 'positive' or 'negative'.

Examples:
Feedback: "I love this amazing phone"
Sentiment: positive

Feedback: "This is the worst and a terrible device ever"
Sentiment: negative

Now classify:
Feedback: "{feedback}"

Return ONLY a valid JSON object like this:
{{"sentiment": "positive"}} OR {{"sentiment": "negative"}}
""",
    input_variables=["feedback"],
)


classifier_chain = prompt1 | model | parser2



prompt2 = PromptTemplate(
    template='Write an appropriate response to this postive feedback \n {feedback}',
    input_variables=['feedback']
)


prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)




branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'Positive' , prompt2 | model | parser),
    (lambda x:x['sentiment'] == 'Negative' , prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)


chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : 'This is a terrible phone'})

print(result)


