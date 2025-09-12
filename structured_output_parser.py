from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain.output_parsers import StructuredOutputParser ,ResponseSchema



model_path = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"
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

schemas = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)

template = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Output ONLY valid JSON with these exact keys: fact_1, fact_2, fact_3.\n"
        "Do not add extra text, explanations, or markdown fences.\n\n"
        "Topic: {topic}\n"
        "{format_instruction}"
    ),
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)



chain = template | model | parser

result = chain.invoke({'topic' : 'Black Hole'})

print(result)