from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",  # open model
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the Capital of indiA")

print(result.content)

