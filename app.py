from flask import Flask, render_template, render_template_string, request
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
import os
#from dotenv import load_dotenv
from pinecone import Pinecone
from langchain import LLMChain
import time
import subprocess
import getpass
import re

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
    api_key=os.getenv("OPENAI_API_KEY"),
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
    api_version="2023-10-01-preview",
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
"""
# HTML template with Jinja2 for dynamic content
HTML_TEMPLATE = 
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>RAG Chatbot Interface</title>
    <div id="logging" style="display: none;"></div>

    <style>
      /* Style adjustments to center video, layout, and buttons */
      body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        display: flex;
        height: 100vh;
        overflow: hidden;
      }

      .left-column {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        box-sizing: border-box;
      }

      #videoContainer {
        position: relative;
        width: 400px;
        height: 600px;
        background: black;
        margin-bottom: 20px;
      }

      #remoteVideo {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }

#follow_up_questions {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    display: grid;
    grid-template-columns: repeat(1, 1fr); /* Two columns for wider buttons */
    grid-auto-rows: auto; /* Adjust height automatically */
    gap: 10px; /* Space between buttons */
    max-width: 100%; /* Ensure container does not overflow */
    justify-items: center; /* Center-align buttons within each grid cell */
}

#follow_up_questions button {
    background-color: #007BFF;
    color: white;
    border: none;
    padding: 8px 10px; /* Increase padding for a larger button size */
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px; /* Slightly larger font */
    text-align: center; /* Center-align button text */
    width: 100%; /* Make button take full width of grid cell */
    box-sizing: border-box;
}

#follow_up_questions button:hover {
    background-color: #0056b3;
}



      #query_form {
        width: 400px;
        display: flex;
        flex-direction: column;
        align-items: center;
      }

      #user_query {
        width: 100%;
        height: 40px;
        font-size: 14px;
        padding: 5px;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-sizing: border-box;
        margin-bottom: 10px;
      }

      #query_form button {
        width: 100%;
        padding: 10px;
        font-size: 14px;
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
      }

      #query_form button:hover {
        background-color: #0056b3;
      }

      .right-column {
        flex: 1;
        padding: 20px;
        overflow-y: auto;
        box-sizing: border-box;
        border-left: 1px solid #ccc;
      }

      .chat-history {
        font-size: 14px;
        line-height: 1.6;
      }

      .chat-history strong {
        display: block;
        margin-bottom: 5px;
      }
    </style>
    <script src="https://aka.ms/csspeech/jsbrowserpackageraw"></script>
    <script src="{{ url_for('static', filename='js/new.js') }}"></script>
    <script>
      // Initialize and start the avatar session automatically on page load
      window.onload = () => {
        startSessionAutomatically();
        
      };
    </script>
    {% if response %}
<script>
  // Call the speak function with the response from the primary LLM
  originalSpeakFunction("{{ response | escapejs }}");
</script>
    {% endif %}
    <script>
      function submitFollowUp(question) {
        document.getElementById('user_query').value = question;
        submitQuery(); // Call AJAX function instead of form submission
      }

      // AJAX submission to prevent page reload
      function submitQuery() {
        const userQuery = document.getElementById('user_query').value;
        if (!userQuery) return; // Do nothing if the input is empty

        fetch('/main', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ user_query: userQuery })
        })
        .then(response => response.json())
        .then(data => {
          // Update chat history with new response
          const chatHistoryDiv = document.getElementById('chatbot_response');
          chatHistoryDiv.innerHTML += `<p><strong>You:</strong> ${userQuery}</p>`;
          chatHistoryDiv.innerHTML += `<p><strong>Avatar:</strong> ${data.response}</p>`;

          // Clear input field
          document.getElementById('user_query').value = '';

          // Update follow-up questions if provided
          const followUpContainer = document.getElementById('follow_up_questions');
          followUpContainer.innerHTML = ''; // Clear old buttons
          if (data.follow_up_questions) {
            data.follow_up_questions.forEach(question => {
              const button = document.createElement('button');
              button.innerText = question;
              button.onclick = () => submitFollowUp(question);
              followUpContainer.appendChild(button);
            });
          }
        })
        .catch(error => {
          console.error('Error:', error);
          alert('An error occurred. Please try again.');
        });
      }
    </script>
  </head>
  <body>
    <!-- Left column for video, buttons, textbox, and submit button -->
    <div class="left-column">
      <div id="videoContainer">
        <div id="overlayArea" style="position: absolute;" hidden="hidden">
          <p id="overlayText" style="font-size: large;">Live Video</p>
        </div>
        <div id="remoteVideo"></div>
        <canvas id="canvas" width="400" height="600" style="background-color: transparent;" hidden="hidden"></canvas>
        <canvas id="tmpCanvas" width="400" height="600" hidden="hidden"></canvas>

        <!-- Overlay generated buttons -->
        <div id="follow_up_questions">
          {% for question in follow_up_questions %}
            <button onclick="submitFollowUp('{{ question }}')">{{ question }}</button>
          {% endfor %}
        </div>
      </div>

      <!-- Textbox and submit button -->
      <form id="query_form" onsubmit="event.preventDefault(); submitQuery();" method="post">
        <textarea id="user_query" name="user_query" placeholder="Enter your question here"></textarea>
        <button type="submit">Submit Query</button>
      </form>
    </div>

    <!-- Right column for chat history -->
    <div class="right-column">
      <h3>Chatbot Response:</h3>
      <div id="chatbot_response" class="chat-history">
        {% for query, answer in chat_history %}
          <p><strong>You:</strong> {{ query }}</p>
          <p><strong>Avatar:</strong> {{ answer }}</p>
        {% endfor %}
      </div>
    </div>
  </body>
</html>
"""


from flask import url_for

from flask import Flask, render_template_string, request, jsonify

from flask import Flask, render_template, render_template_string, request, jsonify

app = Flask(__name__)

# Route for the start page
@app.route('/')
def start_page():
    return render_template('start.html')  # This serves the start page (start.html)

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
    app.run(debug=True, port=5000)