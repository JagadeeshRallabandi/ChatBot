import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configure the Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define paths to the PDF files
PDF_PATHS = ["BTechRules.pdf", "HostelUndertaking.pdf", "SecurityGuidelines.pdf"]

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
                # Debug: Print some extracted text
                print(page_text[:500])  # Print the first 500 characters of the extracted text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    # Debug: Print the number of chunks and a preview
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk preview: {chunks[0][:500]}")
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    # Debug: Verify vector store creation
    print("Vector store created and saved.")


def get_conversational_chain():
    prompt_template = """
    You are the College Chatbot for the National Institute of Technology Andhra Pradesh (NIT AP). As an expert on college-related matters, including academic programs, campus facilities, student services, and college policies, your goal is to provide detailed and accurate responses based on the provided context.

If a question pertains to the name of the institute, respond with: 
"The National Institute of Technology Andhra Pradesh, located in Tadepalligudem, is one of the youngest NITs in India."

For all other questions, answer as thoroughly as possible based on the provided context. If the information is not explicitly mentioned in the provided context, provide a general answer or direct the user to where they might find more information. Always aim to assist and guide the user to the best of your ability.

Context:
{context}

Question:
{question}

Answer:

    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    # Debug: Print the raw response
    print("Raw response:", response)
    st.write("Reply:", response.get("output_text", "No response generated"))


def main():
    st.set_page_config(page_title="College Chatbot", layout="wide")
    st.header("Chat with College Documents using Gemini")
    
    user_question = st.text_input("Ask a Question about the College")
    
    if user_question:
        user_input(user_question)
    
    # Initialization and processing of the PDFs
    with st.sidebar:
        st.title("Menu")
        if st.button("Initialize Chatbot"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(PDF_PATHS)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Chatbot initialized with college documents!")

if __name__ == "__main__":
    main()
