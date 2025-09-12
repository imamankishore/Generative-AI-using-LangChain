from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_path = r"C:\Users\itsam\LangChain Models\gemma-2-2b-it"

device = 0 if torch.cuda.is_available() else -1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": device},   # Accelerate handles device placement
    dtype=torch.float16,       # <- use dtype instead of torch_dtype
)


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Wrap into LangChain
llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name , age , city of the fitional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)