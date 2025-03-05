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

import os
from flask import Flask, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

generated_buttons = []

# Initialize Pinecone client

pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
# Initialize RAG-related components
embeddings = AzureOpenAIEmbeddings(
    deployment="VARELab-TxtEmbeddingLarge",
    model="text-embedding-3-large",
    #api_key=os.getenv("OPENAI_API_KEY"),
    api_key=os.environ.get('AZURE_OPENAI_VARE_KEY'),
    openai_api_version="2023-05-15",
    azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
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
    azure_deployment="VARELab-GPT4o",
    api_key=os.environ.get('AZURE_OPENAI_VARE_KEY'),
    api_version="2024-08-01-preview",
    azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
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
SPEECH_KEY = os.environ.get('AZURE_SPEECH_KEY')
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
            # Get the metadata directly from the document
            metadata = result.get('metadata', {})
            
            # Ensure we have the core metadata fields from Cosmos DB
            if metadata:
                # Add any additional metadata from the result
                metadata['source_type'] = result.get('source_type', 'unknown')
                metadata['score'] = result.get('similarity', 0)
                metadata['category_id'] = result.get('category_id', self.category_id)
                
                # Print the metadata for debugging
                print(f"[DEBUG] Document metadata before passing to Document: {metadata}")
                
                # Create document with metadata
                doc = Document(
                    page_content=result['text'],
                    metadata=metadata
                )
            else:
                # Create a basic document with minimal metadata if none exists
                doc = Document(
                    page_content=result['text'],
                    metadata={
                        'source': result.get('source_type', 'unknown'),
                        'score': result.get('similarity', 0),
                        'category_id': result.get('category_id', self.category_id)
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
            import traceback
            error_traceback = traceback.format_exc()
            print(f"[DETAILED ERROR] {error_traceback}")
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
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Get the index dimension first
        index_info = pc.describe_index(custom_index_name)
        index_dimension = index_info.dimension
        print(f"[DEBUG] Index dimension is {index_dimension}")
        
        # Create the appropriate zero vector
        zero_vector = [0] * index_dimension
        
        # Query all vectors with the correct dimension
        index = pc.Index(custom_index_name)
        response = index.query(
            vector=zero_vector,
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

        intro_prompt = "You are a virtual avatar, designed to help provide insightful information and answer queries formally and in an inviting nature.\n\n"

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
            azure_deployment="VARELab-GPT4o",
            api_key=os.environ.get('AZURE_OPENAI_VARE_KEY'),
            api_version="2024-08-01-preview",
            azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
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
            
        chat_session = session_data[session_id]

        # Process query
        result = chat_session.qa_chain({
            "question": user_query,
            "chat_history": chat_session.chat_history
        })

        response = result['answer']
        source_documents = result.get("source_documents", [])

        # Debug output
        print(f"[DEBUG] Retrieved {len(source_documents)} source documents")

        # Dictionary to consolidate sources by document
        document_pages = {}

        # Process source documents to extract metadata
        for i, doc in enumerate(source_documents):
            print(f"\n[DEBUG] Processing document {i+1}:")
            
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                
                # Extract filename
                filename = None
                if 'source_filename' in metadata:
                    filename = metadata['source_filename']
                elif 'source' in metadata and metadata['source'] != 'unknown':
                    if isinstance(metadata['source'], str) and ('/' in metadata['source'] or '\\' in metadata['source']):
                        filename = os.path.basename(metadata['source'])
                    else:
                        filename = metadata['source']
                
                # Skip if no valid filename
                if not filename or filename == "Unknown Source":
                    continue
                
                # Extract page number
                page_num = None
                if 'page_number' in metadata:
                    page_num = metadata['page_number']
                elif 'page' in metadata:
                    page_num = int(metadata['page']) + 1
                
                # Add to document_pages dictionary
                if filename not in document_pages:
                    document_pages[filename] = set()
                
                if page_num:
                    document_pages[filename].add(page_num)
        
        # Construct HTML citation with hover effect
        if document_pages:
            citation_html = '<span class="citation-container">[Source]<span class="citation-hover">'
            
            # Add each document with its pages
            for filename, pages in document_pages.items():
                if pages:
                    # Sort page numbers
                    sorted_pages = sorted(pages)
                    page_str = "Page" if len(sorted_pages) == 1 else "Pages"
                    pages_formatted = ", ".join(str(p) for p in sorted_pages)
                    citation_html += f"{filename} - {page_str} {pages_formatted}<br>"
                else:
                    citation_html += f"{filename}<br>"
            
            citation_html += '</span></span>'
            
            # Add CSS for hover effect inline (will be included in the response)
            css_style = """
<style>
.citation-container {
    position: relative;
    color: blue;
    text-decoration: underline;
    cursor: pointer;
    display: inline-block;
}
.citation-hover {
    visibility: hidden;
    position: absolute;
    z-index: 1000;
    bottom: 125%;
    left: 0;
    background-color: #333;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 10px;
    opacity: 0;
    transition: opacity 0.3s, visibility 0.3s;
    
    /* Ensure full content display */
    white-space: normal;
    width: max-content;
    max-width: 300px;
    word-wrap: break-word;
    font-weight: normal;
    font-size: 14px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.citation-container:hover .citation-hover {
    visibility: visible;
    opacity: 1;
}
</style>
            """
            
            # Add the citation to the response
            response += " " + css_style + citation_html
        
        # Update chat history with the formatted response
        chat_session.chat_history.append((user_query, response))

        # Return response and updated chat history
        cleaned_response = re.sub(r'^Avatar:\s*', '', response)
        return cleaned_response, chat_session.chat_history

    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        import traceback
        traceback.print_exc()
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
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_VARE_KEY')
AZURE_API_BASE = os.environ.get('AZURE_ENDPOINT')
AZURE_API_VERSION = "2023-10-01-preview"

# Pinecone Configuration
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY2')


HOST = os.environ.get('COSMOS_HOST')
KEY = os.environ.get('COSMOS_KEY')

cosmos_client = CosmosClient(HOST, KEY)

database_name = "varelab-website"
container_name = "ivory-draft"

llm = AzureChatOpenAI(
    azure_deployment="VARELab-GPT4o",
    api_key=os.environ.get('AZURE_OPENAI_VARE_KEY'),
    azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
    api_version="2024-08-01-preview",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)



# Your existing Azure OpenAI embeddings setup
azure_embeddings = AzureOpenAIEmbeddings(
    deployment="VARELab-TxtEmbeddingLarge",
    model="text-embedding-3-large",
    api_key=os.environ.get('AZURE_OPENAI_VARE_KEY'),
    azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
    openai_api_version="2023-05-15",
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
    pinecone_api_key = os.environ.get('PINECONE_API_KEY2')
    pc = Pinecone(api_key=pinecone_api_key)
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
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
                "backgroundColor": "#FFFFFF1A",
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
        SPEECH_ENDPOINT = os.environ.get('SPEECH_ENDPOINT')
        SUBSCRIPTION_KEY = os.environ.get('SPEECH_ENDPOINT')
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
                "backgroundColor": "#00FFFF1A",
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


# Update the similarity search function to include metadata
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
            
            # Use specific fields instead of c.* which is not supported
            query_text = f"""
            SELECT TOP @k c.id, c.text, c.category_id, c.source_type, c.metadata,
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
                print(f"[DEBUG] Raw Cosmos DB result metadata: {json.dumps(item.get('metadata', {}), indent=2)}")
                results.append(item)
            
            print(f"[DEBUG] Found {len(results)} matching documents")
            return results
                
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            raise

async def upload_documents_to_category(docs: list, category_id: str, source_type: str = None):
    """
    Upload documents to a specific category in Cosmos DB with enhanced metadata
    
    Args:
        docs: List of Document objects to upload
        category_id: Category identifier for the documents
        source_type: Source type for the documents (defaults to category_id if not specified)
    """
    if source_type is None:
        source_type = category_id
        
    try:
        print(f"\nUploading documents to category: {category_id}...")
        
        # Ensure all documents have consistent metadata fields
        for doc in docs:
            # Make sure source_filename is present
            if 'source_filename' not in doc.metadata:
                doc.metadata['source_filename'] = "unknown_source"
                
            # Make sure page_number is present for PDFs
            if doc.metadata.get('file_type') == 'pdf' and 'page_number' not in doc.metadata:
                doc.metadata['page_number'] = 1
        
        await upsert_documents_with_embeddings(
            docs,
            azure_embeddings,
            category_id=category_id,
            source_type=source_type
        )
        print(f"Successfully uploaded {len(docs)} documents to category: {category_id}")
        print(f"Document metadata includes: source_filename and page_number (for PDFs)")
    except Exception as e:
        print(f"Error uploading documents to category {category_id}: {str(e)}")
        raise


async def upload_metadata_to_pinecone(metadata_json: dict, embeddings):
    """Upload metadata about the RAG agent to Pinecone"""
    try:
        custom_index_name = "custom-rag-vare"
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        try:
            # Get current index dimension if it exists
            index_info = pc.describe_index(custom_index_name)
            index_dimension = index_info.dimension
            print(f"[DEBUG] Existing index dimension: {index_dimension}")
        except:
            # Index doesn't exist
            index_dimension = 1536  # Default
            
        # Get embedding dimension from a test embedding
        metadata_text = str(metadata_json)
        test_embedding = embeddings.embed_documents([metadata_text])[0]
        embedding_dimension = len(test_embedding)
        print(f"[DEBUG] Embedding dimension: {embedding_dimension}")
        
        # Check if dimensions match, recreate if they don't
        if custom_index_name in existing_indexes and index_dimension != embedding_dimension:
            print(f"[DEBUG] Dimension mismatch. Deleting and recreating index with dimension {embedding_dimension}.")
            pc.delete_index(custom_index_name)
            time.sleep(5)  # Wait for deletion
            existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        if custom_index_name not in existing_indexes:
            print(f"[DEBUG] Creating missing index: {custom_index_name} with dimension {embedding_dimension}")
            pc.create_index(
                name=custom_index_name,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(custom_index_name).status["ready"]:
                time.sleep(1)
                
        custom_index = pc.Index(custom_index_name)
        metadata_vector = test_embedding  # Reuse test embedding
        
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
    """Upload knowledge base documents to Cosmos DB with enhanced metadata"""
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
                    pages = loader.load_and_split()
                    
                    # Enhance PDF metadata with filename and page numbers
                    for page in pages:
                        # PyPDFLoader sets page metadata with zero-indexing, add 1 for human-readable
                        page_num = page.metadata.get('page', 0) + 1
                        # Add enhanced metadata
                        page.metadata.update({
                            'source_filename': filename,
                            'page_number': page_num,
                            'file_type': 'pdf'
                        })
                else:
                    loader = TextLoader(file_path)
                    docs = loader.load_and_split()
                    
                    # Add metadata for text files (no page numbers)
                    for doc in docs:
                        doc.metadata.update({
                            'source_filename': filename,
                            'file_type': 'text'
                        })
                    pages = docs
                
                all_documents.extend(pages)
                
                # Clean up the temporary file
                os.remove(file_path)
                
            except Exception as e:
                print(f"[ERROR] Failed to process file {file.filename}: {e}")
                # Clean up if file was saved
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
                return False, f"Failed to process file {file.filename}: {str(e)}"
        
        # Upload documents to Cosmos DB using our existing function with enhanced metadata
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

        # Your existing code
        print("Starting upload process")
        
        # Add logging throughout the function
        files = request.files.getlist('files')
        print(f"Received {len(files)} files")
        # Get form data and files

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
            deployment="VARELab-TxtEmbeddingLarge",
            model="text-embedding-3-large",
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version="2023-05-15",
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
            "background": request.form.get('backgroundVersion'),
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
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[DETAILED ERROR] {error_traceback}")
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
def get_avatars():
    try:
        custom_index_name = "custom-rag-vare"
        pinecone_api_key = os.environ.get('PINECONE_API_KEY') 
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Get the index info to determine dimension
        index_info = pc.describe_index(custom_index_name)
        index_dimension = index_info.dimension
        print(f"[DEBUG] Index dimension is {index_dimension}")
        
        # Get the index
        index = pc.Index(custom_index_name)
        
        # Create zero vector with correct dimension
        zero_vector = [0] * index_dimension
        
        # Query all vectors
        response = index.query(
            vector=zero_vector,
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
                    'category_id': metadata_dict.get('index_name', ''),
                    'background': metadata_dict.get('background', 'home.png')
                }
                avatars.append(avatar_info)
            except Exception as e:
                print(f"Error processing match: {e}")
                continue
        
        print(f"Found {len(avatars)} avatars")
        return jsonify({'avatars': avatars})
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch avatars: {e}")
        return jsonify({'error': str(e)}), 500
    


@app.route('/config')
def get_config():
    return jsonify({
        "azureSpeechRegion": os.getenv("AZURE_SPEECH_REGION"),
        "azureSpeechSubscriptionKey": os.getenv("AZURE_SPEECH_SUBSCRIPTION_KEY"),
        "ttsVoiceName": os.getenv("TTS_VOICE_NAME"),
        "talkingAvatarCharacterName": os.getenv("TALKING_AVATAR_CHARACTER"),
        "personalVoiceSpeakerProfileID": os.getenv("PERSONAL_VOICE_SPEAKER_PROFILE"),
        "customVoiceEndpointId": os.getenv("CUSTOM_VOICE_ENDPOINT_ID"),
        "customVendpoinIDt": os.getenv("CUSTOM_VOICE_DEPLOYMENT_ID")
    })

########################################################################################################
# R2 chat upload
########################################################################################################

from flask import Flask, request, jsonify
import boto3
from botocore.config import Config
import logging
import json
import os

# R2 Storage Service
class S3Service:
    def __init__(self, s3_client, bucket):
        self.s3_client = s3_client
        self.bucket = bucket

    def upload_json_to_r2(self, key, json_data, content_type='application/json'):
        if isinstance(json_data, dict):
            json_data = json.dumps(json_data)
        
        json_bytes = json_data.encode('utf-8')
        
        # Log the details for debugging
        logging.info(f"Uploading to bucket: {self.bucket}, key: {key}")
        
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json_bytes,
            ContentType=content_type
        )

def new_r2_service():
    # Use your hardcoded values for now since we're not using environment variables
    account = "df2ea43a01264cce57ae81dc60d2d4f5"
    access_key = "8f3319de2021e17b5c5dd67ca810f365"
    secret_key = "54112572aa5f4602b3c32a2cf2cc887a02e2aabc4cd287f2b908263269629ec1"
    bucket = "dialogue-json"
    
    logging.info(f"Initializing R2 service for account: {account[:4]}... and bucket: {bucket}")
    
    r2_config = Config(
        s3={
            "addressing_style": "virtual",
        },
        retries = {
            'max_attempts': 10,
            'mode': 'standard'
        }
    )
    
    s3_client = boto3.client(
        's3',
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=r2_config
    )
    
    return S3Service(s3_client, bucket)

# Add the R2 upload route directly to your app
@app.route('/upload_to_r2', methods=['POST'])
def upload_to_r2():
    try:
        logging.info("Upload to R2 endpoint called")
        
        if request.is_json:
            json_data = request.get_json()
            logging.info(f"Received JSON data: {json.dumps(json_data)[:100]}...")
            
            session_id = json_data.get('sessionId', 'unknown')
            timestamp = json_data.get('timestamp', '')
            file_key = f"chat_{session_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
            
            logging.info(f"Generated file key: {file_key}")
            
            s3_service = new_r2_service()
            s3_service.upload_json_to_r2(file_key, json_data)
            
            return jsonify({"success": True, "message": "Chat data uploaded to R2 successfully", "file_key": file_key}), 200
        else:
            logging.warning("No JSON data found in request")
            return jsonify({"success": False, "message": "No JSON data found in request"}), 400
    except Exception as e:
        logging.error(f"Failed to upload JSON data: {str(e)}", exc_info=True)
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500
    # This function should be called in your main app.py file
# Example: add_r2_routes(app)

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