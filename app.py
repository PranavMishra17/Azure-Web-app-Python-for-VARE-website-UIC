from flask import Flask, render_template, render_template_string, request
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
import os
#from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.chains import LLMChain
import time
import subprocess
import getpass
import re
from pinecone import ServerlessSpec

#load_dotenv()

app = Flask(__name__)

generated_buttons = []

# Initialize Pinecone client
#if not os.getenv("PINECONE_API_KEY"):
    #os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")
#pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc = Pinecone(api_key='9b4c63f4-a0ca-464c-b230-674ead51a686')
# Initialize RAG-related components
embeddings = AzureOpenAIEmbeddings(
    deployment="AzureAdaLangchain",
    model="text-embedding-ada-002",
    #api_key=os.getenv("OPENAI_API_KEY"),
    api_key="370e4756680d40a9978934a4f8af3ed9",
    openai_api_version="2023-10-01-preview",
    azure_endpoint="https://testopenaisaturday.openai.azure.com/",
    openai_api_type="azure",
    chunk_size=512
)

index_name = "langchain-test-index"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

index_name_Q = "ivory-q-index"
existing_indexes_Q = [index_info["name"] for index_info in pc.list_indexes()]
if index_name_Q not in existing_indexes_Q:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Initialize the Pinecone Vector Store
index_Q = pc.Index(index_name_Q)
vectorstore_Q = PineconeVectorStore(index=index_Q, embedding=embeddings, text_key="text")
retriever_Q = vectorstore_Q.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the primary LLM for answering the user's query
LLM_Primary = AzureChatOpenAI(
    azure_deployment="varelabsAssistant",
    api_key="370e4756680d40a9978934a4f8af3ed9",
    api_version="2023-10-01-preview",
    azure_endpoint="https://testopenaisaturday.openai.azure.com/",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Create the condense prompt for the primary LLM
CONDENSE_PROMPT_PRIMARY = PromptTemplate.from_template("""
You are an assistant helping to condense a follow-up question in a conversation between a parent and an Avatar (doctor) about the child's dental hygiene. Use the chat history to rephrase the parent's follow-up question into a standalone question that includes references to any people mentioned.

Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone Question:
""")

# Create the QA prompt for the primary LLM
QA_PROMPT_PRIMARY = PromptTemplate.from_template("""
You are an Avatar (doctor) speaking to a parent about their child's dental hygiene. You must strictly adhere to the script provided below. Use only the information and responses from the script. When appropriate, you can use exact lines from the script, but you may also paraphrase to maintain clarity and coherence. Use the chat history to understand who you are talking to and refer to the individuals appropriately. Provide direct answers to the parent's questions and comments, and talk like a medical professional. Do not include any information not present in the script. Do not ask any follow-up questions unless the parent explicitly asks for more information.

Script Details:
{context}

Conversation History:
{chat_history}

Parent's Question: {question}
Avatar's Response:
""")

# Initialize the Conversational Retrieval Chain for the primary LLM
qa_chain_primary = ConversationalRetrievalChain.from_llm(
    llm=LLM_Primary,
    retriever=retriever,
    condense_question_prompt=CONDENSE_PROMPT_PRIMARY,
    combine_docs_chain_kwargs={'prompt': QA_PROMPT_PRIMARY},
    return_source_documents=True,
    verbose=False
)

# Example usage
chat_history = []

def clean_questions(questions):
    """
    Removes leading numbers and other non-alphabetic characters from each question.
    
    Args:
    - questions (list of str): List of questions with potential leading numbers.
    
    Returns:
    - list of str: Cleaned list of questions without leading numbers or symbols.
    """
    cleaned_questions = [re.sub(r'^\d+[\.\)]?\s*', '', question) for question in questions]
    return cleaned_questions


from flask import url_for

from flask import Flask, render_template, render_template_string, request, jsonify
import uuid
import azure.cognitiveservices.speech as speechsdk
import os

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

# Azure Speech Configuration
SPEECH_KEY = "18f978cca70246309254196a93ce34b4"
SERVICE_REGION = "eastus"
VOICE_NAME = "drdavidNeural"

# Path to store synthesized audio files
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_FOLDER, exist_ok=True)


app = Flask(__name__)

# Route for the start page
@app.route('/')
def start_page():
    return render_template('start.html')  # This serves the start page (start.html)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')



@app.route('/static/<path:filename>')
def serve_audio(filename):
    return send_from_directory(STATIC_FOLDER, filename, mimetype='audio/mpeg')


# Route for the main page
@app.route('/main', methods=['GET', 'POST'])
def main_page():
    global chat_history
    follow_up_questions = []
    response = None

    if request.method == 'POST':
        data = request.get_json()
        user_query = data.get('user_query') if data else None

        if user_query:
            try:
                # Example logic to process user query
                result_primary = qa_chain_primary.apply([{"question": user_query, "chat_history": chat_history}])[0]
                response = result_primary['answer']
                chat_history.append((user_query, response))

                cleaned_response = re.sub(r'^Avatar:\s*', '', response)

                top_chunks = retriever_Q.get_relevant_documents(response)
                follow_up_questions = clean_questions([chunk.page_content for chunk in top_chunks])

                print("Follow-Up Questions:", follow_up_questions)
                
                return jsonify({"response": cleaned_response, "follow_up_questions": follow_up_questions})
            except Exception as e:
                print(f"Error processing query: {e}")
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "No user query provided"}), 400

    # Serve the index.html page for GET requests
    return render_template('index.html', response=None, follow_up_questions=follow_up_questions, chat_history=chat_history)

if __name__ == '__main__':
    #app.run(debug=True, host ='0.0.0.0', port=8000)
    app.run()