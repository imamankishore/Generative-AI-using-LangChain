# # from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# # from langchain_core.prompts import PromptTemplate
# # from langchain_core.output_parsers import StrOutputParser

# # model_name = r"C:\Users\itsam\LangChain Models\TinyLlama-1.1B-Chat-v1.0"


# # llm = HuggingFacePipeline.from_model_id(
# #     model_id=model_name,     
# #     task="text-generation",
# #     model_kwargs={"temperature": 0.7, "max_length": 256}
# # )

# # model = ChatHuggingFace(llm=llm)


# # #1st Prompt -->  detailed prompt

# # template1 = PromptTemplate(
# #     template='Write a detailed Report on {topic}',
# #     input_variables=['topic']
# # )


# # #2nd Prompt -->  summary prompt

# # template2 = PromptTemplate(
# #     template='WWrite a 5 small line summary on text. /n {text}',
# #     input_variables=['text']
# # )


# # parser = StrOutputParser()

# # chain = template1 | model | parser | template2 | model | parser


# # result = chain.invoke({'topic' : 'black hole'})

# # print(result)







# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# import torch

# model_path = r"C:\Users\itsam\LangChain Models\gemma-2-2b-it"

# torch.cuda.set_device(1)


# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_path,     
#     task="text-generation",
# )

# model = ChatHuggingFace(llm=llm)


# #1st Prompt -->  detailed prompt

# template1 = PromptTemplate(
#     template='Write a detailed Report on {topic}',
#     input_variables=['topic']
# )


# #2nd Prompt -->  summary prompt

# template2 = PromptTemplate(
#     template='WWrite a 5 small line summary on text. /n {text}',
#     input_variables=['text']
# )


# parser = StrOutputParser()

# chain = template1 | model | parser | template2 | model | parser


# result = chain.invoke({'topic' : 'black hole'})

# print(result)


from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_path = r"C:\Users\itsam\LangChain Models\gemma-2-2b-it"

# Always use your only GPU (RTX 3050 = cuda:0)
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

# Prompts
template1 = PromptTemplate(
    template='Write a detailed Report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 small line summary on text.\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

report = (template1 | model | parser).invoke({"topic": "black hole"})

# Step 2: Generate summary of that report
summary = (template2 | model | parser).invoke({"text": report})

print("=== Detailed Report ===\n", report)
print("\n=== 5-Line Summary ===\n", summary) 