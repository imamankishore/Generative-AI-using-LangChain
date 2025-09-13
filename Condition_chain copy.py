from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


clf_model_name = "C:/Users/itsam/LangChain Models/distilbert-sst2"
device = 0 if torch.cuda.is_available() else -1

clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_name)
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_model_name)

classifier = pipeline(
    "text-classification",
    model=clf_model,
    tokenizer=clf_tokenizer,
    device=device,
    top_k=1
)


gen_model_name = "C:/Users/itsam/LangChain Models/TinyLlama-1.1B-Chat-v1.0"

gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

gen_pipe = pipeline(
    "text-generation",
    model=gen_model,
    tokenizer=gen_tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.3,
    no_repeat_ngram_size=3
)



llm = HuggingFacePipeline(pipeline=gen_pipe)
parser = StrOutputParser()

def classify_feedback(inputs):
    raw_result = classifier(inputs["feedback"])
    if isinstance(raw_result[0], list):
        result = raw_result[0][0]
    else:
        result = raw_result[0]

    label = result["label"]
    sentiment = "positive" if label.upper() == "POSITIVE" else "negative"
    return {"sentiment": sentiment, "feedback": inputs["feedback"]}

prompt_positive = PromptTemplate(
    template=(
        "You are a customer support agent.\n"
        "Customer feedback (positive): {feedback}\n"
        "Reply warmly and appreciatively. Keep it short, sincere, and friendly.\n"
        "Response:"
    ),
    input_variables=["feedback"],
)


prompt_negative = PromptTemplate(
    template=(
        "You are a customer support agent.\n"
        "Customer feedback (negative): {feedback}\n"
        "Reply politely and empathetically. Keep it short and actionable.\n"
        "Response:"
    ),
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", prompt_positive | llm | parser),
    (lambda x: x["sentiment"] == "negative", prompt_negative | llm | parser),
    RunnableLambda(lambda x: "could not find sentiment"),
)

chain = RunnableLambda(classify_feedback) | branch_chain

result = chain.invoke({"feedback": "This is a terible phone"})
print(result)
