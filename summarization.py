from transformers import pipeline

# summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#  input text
text = """
Artificial intelligence (AI) is a branch of computer science that aims to create machines that can perform tasks that would typically require human intelligence.
These tasks include learning, reasoning, problem-solving, understanding natural language, and perception. AI technologies are being used in a wide range of industries, including healthcare, finance, transportation, and more.
Recent advances in machine learning and deep learning have significantly accelerated the capabilities of AI systems, making them more accurate and efficient.
"""

# Summarize
summary = summarizer(text, max_length=60, min_length=25, do_sample=False)

# Print
print("Summary:\n", summary[0]['summary_text'])
