# faq_chatbot.py

import streamlit as st 
from sentence_transformers import SentenceTransformer 
import numpy as np
import faiss
import openai

def load_faq_data():
    faqs = [
        {"question": "What is your return policy?", "answer": "You can return items within 30 days of purchase."},
        {"question": "How do I track my order?", "answer": "You can track your order using the tracking link sent to your email."},
        {"question": "What payment methods do you accept?", "answer": "We accept credit cards, PayPal, and bank transfers."},
        {"question": "How can I contact customer support?", "answer": "You can contact customer support via email at support@example.com."},
        {"question": "Do you ship internationally?", "answer": "Yes, we ship to many countries worldwide."}
    ]
    return faqs

def create_embeddings(faqs):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    questions = [faq['question'] for faq in faqs]
    embeddings = model.encode(questions)
    return embeddings

def setup_faiss(embeddings):
    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension) 
    index.add(np.array(embeddings).astype('float32')) 
    return index

def get_response(user_query, faqs, index):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([user_query])

    D, I = index.search(np.array(query_embedding).astype('float32'), k=1)
    closest_faq = faqs[I[0][0]]  

    prompt = f":User  {user_query}\nBot: {closest_faq['answer']}\n"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def main():
    st.title("AI Chatbot for FAQs")  

    faqs = load_faq_data()
    embeddings = create_embeddings(faqs)
    index = setup_faiss(embeddings)

    user_input = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        if user_input:
            answer = get_response(user_input, faqs, index)
            st.write(f"**Bot:** {answer}")  
        else:
            st.write("Please enter a question.") 

if __name__ == "__main__":
    main() 
