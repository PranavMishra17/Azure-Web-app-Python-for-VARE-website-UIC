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

from flask import Flask, render_template, url_for
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)



app.debug = True  # Enable debug mode



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

# Add route to serve files from parent directory
@app.route('/sdk/<path:filename>')
def serve_sdk_files(filename):
    parent_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where upload.py is
    return send_from_directory(parent_dir, filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

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
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(file_path)
                
                try:
                    # Extract file content
                    documents = process_file(file_path)
                    for doc in documents:
                        file_contents.append(doc.page_content)
                finally:
                    # Clean up the uploaded file
                    os.remove(file_path)
        
        # Summarize content for prompt generation
        content_summary = "\n".join(file_contents[:3])  # Use the first 3 chunks as a representative sample
        
        # Define prompt generation instructions
        if prompt_type == 'condense':
            messages = [
                (
                    "system",
                    """You are an expert at creating prompts for conversational AI. 
                    Using the provided content, generate a concise and effective condense prompt 
                    for a chat assistant. This assistant is an avatar whose knowledge base 
                    is entirely derived from the content. The condense prompt should help the assistant 
                    retrieve the most relevant question from the conversation history to maintain 
                    clarity and engagement. The prompt must include placeholders for {chat_history} 
                    and {question} and emphasize retrieval without generating new content. Only respond with the prompt.
                    """
                ),
                (
                    "human",
                    f"Content sample:\n{content_summary}\n\nGenerate a condense prompt for the assistant."
                ),
            ]
        elif prompt_type == 'qa':
            messages = [
                (
                    "system",
                    """You are an expert at crafting prompts for conversational AI. 
                    Based on the provided content, generate a QA prompt template for a chat assistant. 
                    The assistant acts as an avatar whose knowledge base comes exclusively from the content. 
                    The QA prompt must guide the assistant to provide clear, concise, and accurate answers 
                    while staying strictly within the knowledge base. Include placeholders for {context}, 
                    {chat_history}, and {question}. Ensure the assistant avoids fabricating information 
                    and maintains engagement by suggesting relevant follow-up questions.
                    """
                ),
                (
                    "human",
                    f"Content sample:\n{content_summary}\n\nGenerate a QA prompt for the assistant."
                ),
            ]
        else:
            return jsonify({'error': 'Invalid prompt type specified'}), 400

        # Invoke the LLM to generate the prompt
        ai_msg = llm.invoke(messages)
        
        return jsonify({'prompt': ai_msg.content})
    
    except Exception as e:
        return jsonify({'error': f'Failed to generate prompt: {str(e)}'}), 500
    

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



def process_file(file_path):
    """Process a single file and return its contents and token count"""
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
        'page_count': len(documents)
    }

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

@app.route('/')
def index():
    """Serve the home page file"""
    return render_template('avatar-page.html')

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
    
    print("Starting server...")
    print("Access the upload form at: http://localhost:5000/")
    app.run(debug=True, port=5000)