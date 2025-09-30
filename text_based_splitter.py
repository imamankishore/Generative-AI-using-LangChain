from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader(r"C:\Users\itsam\LangChain Models\Documents\AMAN_KISHORE_RESUME_GGU.pdf")
# docs = loader.load()


text = """
Artificial Intelligence, often abbreviated as AI, stands as one of the most revolutionary forces of our time, reshaping how we live, work, and interact with technology. It is the science and engineering of creating machines that can think, learn, and act in ways that mimic human intelligence. At its core, AI seeks to replicate human cognitive functions such as perception, reasoning, problem-solving, and decision-making. From its early theoretical foundations to its modern-day real-world implementations, AI has evolved from a niche concept into a transformative power driving innovation across industries. It began as an idea rooted in philosophy and mathematics, with early thinkers dreaming of mechanical reasoning, but today it has grown into a sophisticated discipline powered by data, algorithms, and computing advancements. The modern era of AI is fueled by exponential growth in computing power and vast datasets, allowing machines to learn from experience and improve their performance over time without being explicitly programmed.
Machine Learning, one of AI’s most influential branches, empowers systems to recognize patterns, analyze vast amounts of data, and make predictions based on previous outcomes. It enables computers to learn from past experiences just as humans do, refining their understanding and improving with every iteration. Within Machine Learning lies Deep Learning, which uses multi-layered artificial neural networks inspired by the human brain. These deep networks can identify subtle patterns in data, enabling breakthroughs in image recognition, speech understanding, and natural language generation. Another crucial subset of AI is Natural Language Processing (NLP), which bridges the communication gap between humans and machines. NLP allows AI to read, interpret, and respond to human language with nuance, facilitating applications like chatbots, voice assistants, translation tools, and sentiment analysis systems that understand context, tone, and emotion.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0,
)


result = splitter.split_text(text)

print(result[0])

