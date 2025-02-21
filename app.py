from flask import Flask, render_template, request
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
import os
from pinecone import Pinecone
import time
import re
from pinecone import ServerlessSpec

from flask import session
import uuid

from flask import Flask, Blueprint, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import AzureOpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import AzureChatOpenAI
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.cosmos.aio import CosmosClient
import uuid, time, json, requests, os

from flask import Flask, render_template, url_for, request, redirect, session
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# Session-specific storage
chat_histories = {}
active_sessions = {}  # Will store chat histories and follow-ups per session ID
pending_follow_ups = {}  # Store follow-up questions per session

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


from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for

# Azure Speech Configuration
SPEECH_KEY = "18f978cca70246309254196a93ce34b4"
SERVICE_REGION = "eastus"
VOICE_NAME = "drdavidNeural"

# Path to store synthesized audio files
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_FOLDER, exist_ok=True)



app = Flask(__name__)


app.secret_key = os.urandom(32) # Secret key for session management


##############################################################
# COSMOS DB CONTENT

import re
from langchain.schema import BaseRetriever, Document
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Callable
import asyncio
from pydantic import BaseModel, Field
from pydantic.functional_validators import SkipValidation

# Custom Retriever Class for Cosmos DB
class CosmosDBRetriever(BaseRetriever, BaseModel):
    search_function: SkipValidation[Callable] = Field(...)
    category_id: str = Field(...)
    
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Run the coroutine in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the async function and get results
            results = loop.run_until_complete(
                self.search_function(query, self.category_id)
            )
        finally:
            loop.close()
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result['text'],
                metadata={
                    'source': result.get('source_type', 'unknown'),
                    'score': result['similarity']
                }
            )
            documents.append(doc)
        return documents


from typing import Dict, Tuple, Optional
import asyncio
from datetime import datetime
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Global session storage
session_data = {}

class ChatSession:
    def __init__(self, session_id: str, category_id: str):
        self.session_id = session_id
        self.category_id = category_id
        self.chat_history = []
        self.qa_chain = None
        self.qa_prompt = None
        self.initialized = False


@app.route('/initialize_chat', methods=['POST'])
def initialize_chat():
    # Clear any existing session data for this user
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in session_data:
            del session_data[session_id]
    
    # Create new session ID
    session['session_id'] = str(uuid.uuid4())
    
    data = request.get_json()
    category_id = data.get('categoryId')
    
    if category_id:
        try:
            # Initialize new chat session
            initialize_chat_session(session['session_id'], category_id)
            return jsonify({"status": "success", "session_id": session['session_id']})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "No category ID provided"}), 400

def initialize_chat_session(session_id: str, category_id: str) -> ChatSession:
    """Initialize a new chat session with necessary components"""
    if session_id in session_data:
        return session_data[session_id]

    try:
        session = ChatSession(session_id, category_id)
        
        # Query Pinecone for the QA prompt
        custom_index_name = "custom-rag-vare"
        pinecone_api_key = "9b4c63f4-a0ca-464c-b230-674ead51a686"
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(custom_index_name)
        
        # Query all vectors to find matching category
        response = index.query(
            vector=[0] * 1536,
            top_k=100,
            include_metadata=True
        )

        # Find the matching record
        matching_prompt = None
        for match in response['matches']:
            try:
                metadata_text = match['metadata']['text']
                metadata_dict = eval(metadata_text)
                if metadata_dict.get('index_name') == category_id:
                    matching_prompt = metadata_dict['prompts']['qa_prompt']
                    break
            except Exception as e:
                print(f"Error processing match: {e}")
                continue

                    # Create default prompt if none found or if the found prompt doesn't have required variables
        default_prompt = """Use the following context to answer the question. If you cannot answer from the context, say you don't have enough information. Answer in 2-4 sentences only. No long answers.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:"""

        intro_prompt = "You are a virtual avatar, designed to provide insightful information and answer queries.\n\n"

        if matching_prompt:
            # Verify the prompt has all required variables
            if all(var in matching_prompt for var in ["{context}", "{chat_history}", "{question}"]):
                # Append the sentence limit instruction to the prompt
                qa_prompt_template = intro_prompt + matching_prompt + "\n\n Answer in 2-3 sentences only. No long answers."
            else:
                print(f"WARNING: Prompt for category {category_id} missing required variables. Using default prompt.")
                qa_prompt_template = intro_prompt + matching_prompt + default_prompt
        else:
            qa_prompt_template = default_prompt

        if matching_prompt:
            # Create two separate prompts - one for condensing and one for QA
            condense_prompt = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

            qa_prompt = PromptTemplate(
                template=qa_prompt_template,
                # Make sure all these variables are available in the template
                input_variables=["context", "chat_history", "question"]
            )
        else:
            raise ValueError(f"No matching prompts found for category_id: {category_id}")

        # Initialize components
        cosmos_retriever = CosmosDBRetriever(
            search_function=similarity_search_by_category,
            category_id=category_id
        )

        llm = AzureChatOpenAI(
            azure_deployment="varelabsAssistant",
            api_key="370e4756680d40a9978934a4f8af3ed9",
            api_version="2023-10-01-preview",
            azure_endpoint="https://testopenaisaturday.openai.azure.com/",
            temperature=0.5
        )

        # Modified QA chain initialization
        session.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=cosmos_retriever,
            condense_question_prompt=condense_prompt,  # Use separate condense prompt
            combine_docs_chain_kwargs={
                'prompt': qa_prompt,
                'document_variable_name': 'context'  # Explicitly set this
            },
            return_source_documents=True,
            verbose=True
        )

        session.initialized = True
        session_data[session_id] = session
        return session

    except Exception as e:
        print(f"Error initializing chat session: {str(e)}")
        raise

def process_chat(session_id: str, category_id: str, user_query: str):
    """Process a chat message and return response"""
    try:
        # Simply get the existing session
        if session_id not in session_data:
            raise ValueError("Chat session not initialized. Please start a new conversation.")
            
        session = session_data[session_id]

        # Process query
        result = session.qa_chain({
            "question": user_query,
            "chat_history": session.chat_history
        })

        response = result['answer']
        session.chat_history.append((user_query, response))
        
        cleaned_response = re.sub(r'^Avatar:\s*', '', response)
        return cleaned_response, session.chat_history

    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}", []
    

@app.route('/main2', methods=['GET', 'POST'])
def main_page2():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    if request.method == 'POST':
        data = request.get_json()
        user_query = data.get('user_query')
        category_id = data.get('categoryId')

        if user_query and category_id:
            response, chat_history = process_chat(
                session_id=session_id,
                category_id=category_id,
                user_query=user_query
            )
            
            return jsonify({
                "response": response,
                "chat_history": chat_history
            })

    return render_template('index.html', 
                         session_id=session_id,
                         response=None,
                         chat_history=[])


#################################################

# upload.py content



UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Ensure Flask app config has this value

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = "370e4756680d40a9978934a4f8af3ed9"
AZURE_API_BASE = "https://testopenaisaturday.openai.azure.com/"
AZURE_API_VERSION = "2023-10-01-preview"

# Pinecone Configuration
PINECONE_API_KEY = "pcsk_25DwYj_auXVzzL4Erdbn7SxrNiuvAbSvEmvaD2pYGvZwR9W58G7rQbCQzrfPYk5RjFmYM"


HOST = "https://varelab-website.documents.azure.com:443/"
KEY = "hTzGghBPVfYAqSYrMMrQEnkYB2HSZr0ySo1E8BXl6nchZUgpmPTfBsJ9MJVxvVdeP0aOgojvqQMuACDbrWwhpw=="

cosmos_client = CosmosClient(HOST, KEY)

database_name = "varelab-website"
container_name = "ivory-draft"

llm = AzureChatOpenAI(
    azure_deployment="varelabsAssistant",
    api_key="370e4756680d40a9978934a4f8af3ed9",
    api_version="2023-10-01-preview",
    azure_endpoint="https://testopenaisaturday.openai.azure.com/",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Your existing Azure OpenAI embeddings setup
azure_embeddings = AzureOpenAIEmbeddings(
    deployment="AzureAdaLangchain",
    model="text-embedding-ada-002",
    api_key="370e4756680d40a9978934a4f8af3ed9",
    openai_api_version="2023-10-01-preview",
    azure_endpoint="https://testopenaisaturday.openai.azure.com/",
    openai_api_type="azure",
    chunk_size=512
)


###########################################################################################



@app.route('/index')
def index_page():
    session['session_id'] = str(uuid.uuid4())  # Always assign a new session ID
    return render_template('index.html', session_id=session['session_id'])





@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/static/<path:filename>')
def serve_audio(filename):
    return send_from_directory(STATIC_FOLDER, filename, mimetype='audio/mpeg')

def cleanup_old_sessions(max_age=3600):  # max_age in seconds (1 hour)
    current_time = time.time()
    for session_id in list(chat_histories.keys()):
        if current_time - session.get(f'last_access_{session_id}', 0) > max_age:
            del chat_histories[session_id]
            del pending_follow_ups[session_id]  # Also clean up follow-ups

# Schedule periodic cleanup (you can call this periodically using a scheduler)
def periodic_cleanup():
    cleanup_old_sessions()

    
@app.route('/')
def index():
    return render_template('avatar-page.html')

@app.route('/start', methods=['GET', 'POST'])
def start_page():
    if request.method == 'POST':
        user_id = request.form.get('user_id') or request.args.get('user_id', "unknown")
        session['user_id'] = user_id
        session['session_id'] = str(uuid.uuid4())
        return redirect(url_for('main_page', session_id=session['session_id'], user_id=session['user_id']))

    user_id = request.args.get('user_id')
    return render_template('start.html', session_id=session.get('session_id', ''), user_id=user_id)

@app.route('/main', methods=['GET', 'POST'])
def main_page():
    # Get or create session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        print(f"[DEBUG] New session created: {session['session_id']}")
    
    session_id = session['session_id']
    user_id = session.get('user_id', 'unknown')

    # Debugging
    print(f"Session ID: {session_id}")
    print(f"User ID: {user_id}")
    
    # Initialize if not exists
    if session_id not in chat_histories:
        chat_histories[session_id] = []
        pending_follow_ups[session_id] = []

    if request.method == 'POST':
        data = request.get_json()
        user_query = data.get('user_query') if data else None

        if user_query:
            try:
                # Process user query with session-specific chat history
                result_primary = qa_chain_primary.apply([{
                    "question": user_query, 
                    "chat_history": chat_histories[session_id]
                }])[0]
                
                response = result_primary['answer']
                chat_histories[session_id].append((user_query, response))

                cleaned_response = re.sub(r'^Avatar:\s*', '', response)

                # Generate follow-up questions for this session
                top_chunks = retriever_Q.get_relevant_documents(response)
                follow_up_questions = clean_questions([chunk.page_content for chunk in top_chunks])
                
                # Save follow-up questions for this session
                pending_follow_ups[session_id] = follow_up_questions
                print(f"Follow-Up Questions Updated for session {session_id}:", follow_up_questions)
                
                # Update last access time for session cleanup
                session[f'last_access_{session_id}'] = time.time()
                
                return jsonify({"response": cleaned_response})
            except Exception as e:
                print(f"Error processing query: {e}")
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "No user query provided"}), 400

    # For GET requests, pass all required parameters including session_id
    return render_template('index.html', 
                         session_id=session_id, 
                          user_id=user_id, # Add this line
                         response=None, 
                         follow_up_questions=pending_follow_ups.get(session_id, []),
                         chat_history=chat_histories.get(session_id, []))

# Remove the separate index route as it's not needed
# The main_page route now handles everything we need

@app.route('/get_follow_ups', methods=['GET'])
def get_follow_up_questions():
    """Serve follow-up questions for the current session."""
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"follow_up_questions": []})
    
    return jsonify({
        "follow_up_questions": pending_follow_ups.get(session_id, [])
    })




##############################################################################

# Upload.py content

# Add route to serve files from parent directory
@app.route('/sdk/<path:filename>')
def serve_sdk_files(filename):
    parent_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where upload.py is
    return send_from_directory(parent_dir, filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

def process_file(file_path):
    """Process a single file and return its contents and metadata"""
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:  # txt file
            loader = TextLoader(file_path)
        
        documents = loader.load()
        total_text = ' '.join([doc.page_content for doc in documents])
        # Approximate token count (rough estimation: 1 token ≈ 4 characters)
        token_count = len(total_text) / 4
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)
        
        return {
            'chunks': chunks,
            'token_count': token_count,
            'page_count': len(documents),
            'success': True
        }
    except Exception as e:
        print(f"[ERROR] Failed to process file {file_path}: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
    
    
@app.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    try:
        # Ensure files are uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Check prompt type
        prompt_type = request.form.get('prompt_type')
        if not prompt_type:
            return jsonify({'error': 'Prompt type not specified'}), 400
        
        # Process uploaded files to extract content
        files = request.files.getlist('files')
        file_contents = []
        temp_files = []  # Keep track of temporary files
        
        try:
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    temp_files.append(file_path)
                    
                    # Process the file
                    result = process_file(file_path)
                    if not result.get('success', False):
                        raise Exception(f"Failed to process {filename}: {result.get('error', 'Unknown error')}")
                    
                    if 'chunks' in result:
                        for chunk in result['chunks']:
                            file_contents.append(chunk.page_content)
                    else:
                        print(f"Warning: No chunks found in result for {filename}")
                        continue
            
            if not file_contents:
                raise Exception("No valid content extracted from files")
            
            # Summarize content for prompt generation
            content_samples = [content[:1000] for content in file_contents[:3]]
            content_summary = "\n\n---\n\n".join(content_samples)
            
            # Define prompt generation instructions
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert at crafting prompts for conversational AI. 
                    Based on the provided content, generate a QA prompt template for a chat assistant. 
                    The assistant acts as an avatar whose knowledge base comes exclusively from the content. 
                    The QA prompt must guide the assistant to provide clear, concise, and accurate answers 
                    while staying strictly within the knowledge base. Include placeholders for {context}, 
                    {chat_history}, and {question}."""
                },
                {
                    "role": "user",
                    "content": f"Content sample:\n{content_summary}\n\nGenerate a QA prompt for the assistant."
                }
            ]

            # Generate the prompt using the LLM
            response = llm.invoke(messages)
            if not response or not hasattr(response, 'content'):
                raise Exception("Failed to generate prompt from LLM")
            
            return jsonify({'prompt': response.content})
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    except Exception as e:
        print(f"[ERROR] Generate prompt failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_pinecone_index(index_name):
    #pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_api_key = "pcsk_25DwYj_auXVzzL4Erdbn7SxrNiuvAbSvEmvaD2pYGvZwR9W58G7rQbCQzrfPYk5RjFmYM"

    pc = Pinecone(api_key=pinecone_api_key)
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
        # Wait for index to be ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    
    return pc.Index(index_name)

    
@app.route('/estimate_hosting_cost', methods=['POST'])
def estimate_hosting_cost():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Save file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        result = process_file(filepath)
        # Calculate cost: 50 cents per 10 pages
        cost = (result['page_count'] / 10) * 0.50
        
        return jsonify({
            'hosting_cost': round(cost, 2),
            'token_count': int(result['token_count']),
            'page_count': result['page_count']
        })
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/avatar-conv.html')
def avatar_conversation():
    return render_template('avatar-conv.html')

@app.route('/generate_avatar_response', methods=['POST'])
def generate_avatar_response():
    data = request.json
    
    # Get the corresponding voice based on avatar version
    voice_mapping = {
        'max-business': 'en-US-JacobNeural',
        'lisa-casual': 'en-US-NancyNeural',
        # Add other mappings
    }
    
    voice = voice_mapping.get(data['avatarVersion'], 'en-US-JacobNeural')
    
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    
    try:
        # Initialize avatar synthesis
        synthesis_config = {
            "synthesisConfig": {
                "voice": voice,
            },
            "avatarConfig": {
                "customized": False,
                "talkingAvatarCharacter": data['avatarVersion'],
                "talkingAvatarStyle": 'business',
                "videoFormat": "mp4",
                "videoCodec": "h264",
                "subtitleType": "soft_embedded",
                "backgroundColor": "#FFFFFFFF",
            }
        }
        
        # For now, return a simple response
        return jsonify({
            'success': True,
            'message': 'Response generated',
            'synthesis_config': synthesis_config
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/initialize_avatar', methods=['POST'])
def initialize_avatar():
    try:
        data = request.json
        avatar_version = data.get('avatarVersion')
        initial_text = data.get('text', "Hello! I'm your avatar assistant. How can I help you today?")
        
        # Configuration for Azure Speech Service
        SPEECH_ENDPOINT = "https://westus2.api.cognitive.microsoft.com"
        SUBSCRIPTION_KEY = "c897d534a33b4dd7a31e73026200226b"
        API_VERSION = "2024-04-15-preview"
        
        # Get voice based on avatar version
        voice_mapping = {
            'max-business': 'en-US-JacobNeural',
            'lisa-casual': 'en-US-NancyNeural',
            'dr-david-avenetti': 'drdavidNeural',
            'prof-zalake': 'en-US-JacobNeural'
        }
        
        voice = voice_mapping.get(avatar_version, 'en-US-JacobNeural')
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Prepare synthesis request
        url = f'{SPEECH_ENDPOINT}/avatar/batchsyntheses/{job_id}?api-version={API_VERSION}'
        
        headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY
        }
        
        payload = {
            'synthesisConfig': {
                "voice": voice,
            },
            'customVoices': {},
            "inputKind": "plainText",
            "inputs": [
                {
                    "content": initial_text,
                },
            ],
            "avatarConfig": {
                "customized": False,
                "talkingAvatarCharacter": avatar_version,
                "talkingAvatarStyle": 'business',
                "videoFormat": "mp4",
                "videoCodec": "h264",
                "subtitleType": "soft_embedded",
                "backgroundColor": "#FFFFFFFF",
            }
        }
        
        # Submit synthesis job
        response = requests.put(url, json.dumps(payload), headers=headers)
        
        if response.status_code >= 400:
            raise Exception(f"Failed to submit synthesis job: {response.text}")
            
        job_id = response.json()["id"]
        
        # Poll for job completion
        while True:
            status_url = f'{SPEECH_ENDPOINT}/avatar/batchsyntheses/{job_id}?api-version={API_VERSION}'
            status_response = requests.get(status_url, headers={'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY})
            
            if status_response.status_code >= 400:
                raise Exception(f"Failed to get synthesis status: {status_response.text}")
                
            status_data = status_response.json()
            status = status_data['status']
            
            if status == 'Succeeded':
                # Get video URL from the response
                video_url = status_data["outputs"]["result"]
                return jsonify({
                    'success': True,
                    'videoUrl': video_url,
                    'jobId': job_id
                })
            elif status == 'Failed':
                raise Exception("Avatar synthesis failed")
            
            time.sleep(2)  # Poll every 2 seconds
            
    except Exception as e:
        app.logger.error(f"Avatar initialization error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/start_dental')
def start_dental():
    return render_template('start.html')

async def upsert_documents_with_embeddings(documents, embeddings_model, category_id: str, source_type: str):
    """
    Asynchronously upsert documents with embeddings and category information
    """
    async with CosmosClient(HOST, credential=KEY) as client:
        try:
            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)
            
            for doc in documents:
                embedding = embeddings_model.embed_query(doc.page_content)
                
                # Add category information to the document
                cosmos_doc = {
                    "id": doc.metadata.get("id", str(hash(doc.page_content))),
                    "text": doc.page_content,
                    "embedding": embedding,
                    "metadata": doc.metadata,
                    "category_id": category_id,
                    "source_type": source_type,
                    "created_at": datetime.now(timezone.utc).isoformat()

                }
                
                await container.upsert_item(cosmos_doc)
                print(f"Upserted document with ID: {cosmos_doc['id']} for category: {category_id}")
        except Exception as e:
            print(f"Error upserting documents: {str(e)}")
            raise

async def similarity_search_by_category(query: str, category_id: str, k: int = 5):
    """
    Search for similar documents within a specific category using vector search
    """
    async with CosmosClient(HOST, credential=KEY) as client:
        try:
            # Generate embedding for the query
            vector = azure_embeddings.embed_query(query)
            
            # Convert vector to string format for query
            embeddings_string = ','.join(map(str, vector))
            
            # Construct query with proper VectorDistance syntax
            query_text = f"""
            SELECT TOP @k c.id, c.text, c.category_id, c.source_type,
            VectorDistance(c.embedding, [{embeddings_string}], false, {{'dataType': 'float32', 'distanceFunction': 'cosine'}}) AS similarity
            FROM c
            WHERE c.category_id = @category_id
            ORDER BY VectorDistance(c.embedding, [{embeddings_string}], false, {{'dataType': 'float32', 'distanceFunction': 'cosine'}})
            """
            
            parameters = [
                {"name": "@k", "value": k},
                {"name": "@category_id", "value": category_id}
            ]
            
            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)
            
            results = []
            async for item in container.query_items(
                query=query_text,
                parameters=parameters
            ):
                results.append(item)
            
            return results
                
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            raise  # This will help us see the full error stack

async def upload_documents_to_category(docs: list, category_id: str, source_type: str = None):
    """
    Upload documents to a specific category in Cosmos DB
    
    Args:
        docs: List of Document objects to upload
        category_id: Category identifier for the documents
        source_type: Source type for the documents (defaults to category_id if not specified)
    """
    if source_type is None:
        source_type = category_id
        
    try:
        print(f"\nUploading documents to category: {category_id}...")
        await upsert_documents_with_embeddings(
            docs,
            azure_embeddings,
            category_id=category_id,
            source_type=source_type
        )
        print(f"Successfully uploaded {len(docs)} documents to category: {category_id}")
    except Exception as e:
        print(f"Error uploading documents to category {category_id}: {str(e)}")
        raise

# Example usage with PDF reading:
"""
# First read your PDF and convert to Documents
from langchain_community.document_loaders import PyPDFLoader

# Load and split the PDF
loader = PyPDFLoader("your_pdf_path.pdf")
pdf_docs = loader.load_and_split()

# Upload to Cosmos DB
await upload_documents_to_category(
    docs=pdf_docs,
    category_id="ivory_script",
    source_type="pdf_content"
)
"""

async def upload_metadata_to_pinecone(metadata_json: dict, embeddings):
    """Upload metadata about the RAG agent to Pinecone"""
    try:
        custom_index_name = "custom-rag-vare"
        pinecone_api_key = "9b4c63f4-a0ca-464c-b230-674ead51a686"
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if custom_index_name not in existing_indexes:
            print(f"[DEBUG] Creating missing index: {custom_index_name}")
            pc.create_index(
                name=custom_index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(custom_index_name).status["ready"]:
                time.sleep(1)
                
        custom_index = pc.Index(custom_index_name)
        metadata_text = str(metadata_json)
        metadata_vector = embeddings.embed_documents([metadata_text])[0]
        
        # Use index_name as unique identifier
        metadata_id = f"metadata-{metadata_json['index_name']}"
        existing_metadata = custom_index.fetch(ids=[metadata_id])
        
        if metadata_id not in existing_metadata.get("vectors", {}):
            print(f"[DEBUG] Inserting metadata into `custom-rag-vare`")
            custom_index.upsert(vectors=[(metadata_id, metadata_vector, {"text": metadata_text})])
            print(f"[DEBUG] Metadata successfully added to `custom-rag-vare`")
        else:
            print(f"[DEBUG] Metadata for {metadata_id} already exists, skipping insertion.")
            
        return True, "Metadata uploaded successfully"
        
    except Exception as e:
        print(f"[ERROR] Failed to upload metadata: {e}")
        return False, str(e)

async def upload_knowledge_to_cosmos(files: list, category_id: str, embeddings):
    """Upload knowledge base documents to Cosmos DB"""
    try:
        all_documents = []
        
        # Process each file
        for file in files:
            try:
                # Save file temporarily
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Process the saved file
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                    
                pages = loader.load_and_split()
                all_documents.extend(pages)
                
                # Clean up the temporary file
                os.remove(file_path)
                
            except Exception as e:
                print(f"[ERROR] Failed to process file {file.filename}: {e}")
                # Clean up if file was saved
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
                return False, f"Failed to process file {file.filename}: {str(e)}"
        
        # Upload documents to Cosmos DB using our existing function
        await upload_documents_to_category(
            docs=all_documents,
            category_id=category_id,
            source_type="rag_knowledge_base"
        )
        
        return True, "Knowledge base uploaded successfully"
        
    except Exception as e:
        print(f"[ERROR] Failed to upload knowledge base: {e}")
        return False, str(e)

@app.route('/upload_to_rag', methods=['POST'])
async def upload_to_rag():
    try:
        # Get form data and files
        files = request.files.getlist('files')
        avatar_name = request.form.get('avatarName')
        qa_prompt = request.form.get('qaPrompt')
        
        if not all([files, avatar_name, qa_prompt]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Clean and format index/category name
        clean_name = avatar_name.strip().replace(' ', '-').replace('_', '-')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '-').lower()
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        category_id = f"{clean_name}-{timestamp}"
        
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            deployment="AzureAdaLangchain",
            model="text-embedding-ada-002",
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version="2023-10-01-preview",
            azure_endpoint=AZURE_API_BASE,
            openai_api_type="azure",
            chunk_size=512
        )
        
        # Prepare metadata
        metadata_json = {
            "avatar_name": avatar_name,
            "avatar_description": request.form.get('avatarDescription'),
            "avatar_version": request.form.get('avatarVersion'),
            "index_name": category_id,
            "files": [f.filename for f in files],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "api_details": {
                "cosmos_database": "varelab-website",
                "cosmos_container": "ivory-draft",
                "category_id": category_id,
                "azure_deployment": "AzureAdaLangchain",
                "azure_api_base": AZURE_API_BASE
            },
            "prompts": {
                "qa_prompt": qa_prompt
            }
        }
        
        # Upload metadata to Pinecone
        meta_success, meta_message = await upload_metadata_to_pinecone(metadata_json, embeddings)
        if not meta_success:
            return jsonify({'error': f'Failed to upload metadata: {meta_message}'}), 500
            
        # Upload knowledge base to Cosmos DB
        kb_success, kb_message = await upload_knowledge_to_cosmos(files, category_id, embeddings)
        if not kb_success:
            return jsonify({'error': f'Failed to upload knowledge base: {kb_message}'}), 500
        
        # Instead of returning JSON with redirect
        return render_template('avatar-page.html')
        
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

# Add a new route to directly access the avatar page
@app.route('/avatar-page')
def avatar_page():
    return render_template('avatar-page.html')

@app.route('/create_avatar')
def create_avatar():
    return render_template('upload.html')

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.route('/get_avatars')
async def get_avatars():
    try:
        custom_index_name = "custom-rag-vare"
        pinecone_api_key = "9b4c63f4-a0ca-464c-b230-674ead51a686"
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Get the index
        index = pc.Index(custom_index_name)
        
        # Query all vectors
        response = index.query(
            vector=[0] * 1536,  # Dummy vector to get all records
            top_k=100,
            include_metadata=True
        )
        
        avatars = []
        for match in response['matches']:
            try:
                # Parse the metadata string back to dict
                metadata_text = match['metadata']['text']
                # Clean up the string and convert to dict
                metadata_dict = eval(metadata_text)
                
                avatar_info = {
                    'avatar_name': metadata_dict.get('avatar_name', 'Unknown'),
                    'avatar_description': metadata_dict.get('avatar_description', ''),
                    'avatar_version': metadata_dict.get('avatar_version', ''),
                    'category_id': metadata_dict.get('index_name', '')
                }
                avatars.append(avatar_info)
            except Exception as e:
                print(f"Error processing match: {e}")
                continue
        
        print(f"Found {len(avatars)} avatars")  # Debug print
        return jsonify({'avatars': avatars})
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch avatars: {e}")
        return jsonify({'error': str(e)}), 500
    

########################################################################################################


if __name__ == '__main__':

        # Check for required environment variables
    required_vars = {
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        exit(1)
    #app.run(debug=True, host ='0.0.0.0', port=8000)
    app.run()