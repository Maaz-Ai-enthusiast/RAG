# Install Streamlit and required libraries if running for the first time
!pip install streamlit sentence-transformers transformers scikit-learn

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline('text-generation', model='gpt2')

# Load and parse the data
data_path = 'data.txt'  # Path to your Q&A text file

# Function to parse Q&A pairs from the file
def load_data(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    
    qa_pairs = []
    question, answer = None, None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d+\.\s*Q:', line):
            if question and answer:
                qa_pairs.append((question.strip(), answer.strip()))
            question = re.sub(r'^\d+\.\s*Q:\s*', '', line)
            answer = None
        elif line.startswith('A:'):
            answer = line[2:].strip()
    if question and answer:
        qa_pairs.append((question.strip(), answer.strip()))
    return qa_pairs

# Load and embed Q&A data
qa_pairs = load_data(data_path)
questions = [pair[0] for pair in qa_pairs]
question_embeddings = np.array(embedder.encode(questions))

# Define retrieval and generation functions
def retrieve_similar_question(query, top_k=1):
    query_embedding = embedder.encode([query]).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [(qa_pairs[idx][0], qa_pairs[idx][1]) for idx in top_indices]
    return results

def generate_response(query):
    similar_question, answer = retrieve_similar_question(query)[0]
    input_text = f"Question: {query}\nAnswer: {answer}\nFurther explanation:"
    response = generator(input_text, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Streamlit UI
st.title("Retrieval-Augmented Generation (RAG) Demo")
st.write("Ask a question related to data structures and algorithms:")

# Input query
user_query = st.text_input("Enter your question:")

# Display response
if user_query:
    st.write("### Retrieved Answer:")
    similar_question, answer = retrieve_similar_question(user_query)[0]
    st.write(answer)
    
    st.write("### AI-Generated Explanation:")
    generated_text = generate_response(user_query)
    st.write(generated_text)
