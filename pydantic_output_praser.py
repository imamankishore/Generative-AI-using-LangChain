from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re, json   # ✅ added here

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

class Person(BaseModel):
    name : str = Field(description='Name of the person')
    age : int = Field(gt=10 , description='age of the person')
    city : str = Field(description='Name of the city from which the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "You are a JSON generator.\n"
        "Generate a fictional person from {place}.\n\n"
        "⚠️ IMPORTANT: Output only a JSON object with values.\n"
        "Example:\n"
        "{\"name\": \"Aman\", \"age\": 23, \"city\": \"Delhi\"}\n\n"
        "Do NOT output properties, required, type, schema, or explanations.\n"
        "{format_instruction}"
    ),
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)


prompt = template.invoke({'place': 'india'})

result = model.invoke(prompt)

print("Raw Model Output:", result.content)  # <-- debug

# --- Clean the output before parsing ---

match = re.search(r"\{[\s\S]*?\}", result.content, re.DOTALL)
if match:
    cleaned_output = match.group()
else:
    raise ValueError(f"Model output not JSON: {result.content}")

final_result = parser.parse(cleaned_output)

print(final_result)
