# MedRAG Avatar Platform - IVORY

A web application for creating and interacting with custom talking avatars powered by Azure Cognitive Services and RAG (Retrieval Augmented Generation) technology.

![Intro](https://github.com/V-ARE/AIAvatar/blob/a961d8108ee83d88bb2349159dd3ea53a5eb68c3/image_2025-03-17_181028816.png)

## Overview

This project allows users to create and interact with AI-powered talking avatars. Users can upload their own knowledge base documents (PDF, TXT), select avatar appearances, customize backgrounds, and define prompts to create personalized conversational agents. The avatars use speech synthesis and natural language processing to provide dynamic, informative responses based on the uploaded knowledge base.

Youtube Demo: https://youtu.be/tZ5aoUfyKgM 

![Avatar](https://github.com/V-ARE/AIAvatar/blob/a961d8108ee83d88bb2349159dd3ea53a5eb68c3/image_2025-03-17_181156475.png)

## Features

- **Custom Avatar Creation**: Create personalized avatars with uploaded knowledge base
- **Document Processing**: Support for PDF and TXT files
- **Real-time Avatar Interaction**: Web-based interface for conversing with avatars
- **Azure Cognitive Services Integration**: Text-to-speech and talking avatar capabilities
- **Retrieval Augmented Generation (RAG)**: Uses Azure Cosmos DB and vector search for knowledge retrieval
- **WebRTC Streaming**: Real-time audio and video streaming for avatar interactions
- **Responsive Design**: Works across different device sizes

![Upload](https://github.com/V-ARE/AIAvatar/blob/a961d8108ee83d88bb2349159dd3ea53a5eb68c3/image_2025-03-17_181335080.png)

## Technology Stack

- **Backend**: 
  - Flask (Python)
  - Azure OpenAI for text generation
  - Azure Cosmos DB for document storage and vector search
  - Pinecone for vector indexing
  - Azure Speech Services for TTS

- **Frontend**:
  - HTML/CSS/JavaScript
  - WebRTC for real-time communication
  - Azure Speech SDK for browser integration

- **Storage**:
  - Azure Cosmos DB for document storage
  - Pinecone for vector embeddings
  - Cloudflare R2 for chat history storage

## Getting Started

### Prerequisites

- Python 3.8+
- Azure account with:
  - Azure OpenAI API access
  - Azure Cognitive Services (Speech)
  - Azure Cosmos DB
- Pinecone account
- Cloudflare R2 storage (optional, for chat history)

### Environment Variables

Create a `.env` file with the following variables:

```
AZURE_OPENAI_VARE_KEY=your_azure_openai_key
AZURE_ENDPOINT=your_azure_endpoint
PINECONE_API_KEY=your_pinecone_key
PINECONE_API_KEY2=your_second_pinecone_key
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=eastus
COSMOS_HOST=your_cosmos_db_host
COSMOS_KEY=your_cosmos_db_key
```

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```

## Usage

### Creating a Custom Avatar

1. Navigate to the avatar creation page
2. Enter avatar name and description
3. Upload knowledge base documents (PDF, TXT)
4. Select or generate a QA prompt template
5. Choose an avatar character and background
6. Click "Create Avatar"

### Interacting with Avatars

1. Navigate to the avatar gallery page
2. Select an avatar to start a conversation
3. Ask questions via text input or click suggested follow-up questions
4. The avatar will respond with synthesized speech and animation
5. Chat history can be saved for future reference

## Project Structure

- `app.py`: Main Flask application
- `avatar-conv.js`: JavaScript for avatar conversation interface
- `avatar-conv.html`: HTML template for avatar conversation
- `avatar-page.html`: Gallery of available avatars
- `upload.html`: Avatar creation interface
- `static/`: Static files (images, CSS, JS)
- `templates/`: HTML templates

## Features in Detail

### RAG Implementation

The system uses Retrieval Augmented Generation with:
- Document chunking and embedding via Azure OpenAI embeddings
- Storage in Cosmos DB with vector capabilities
- Query-time retrieval based on semantic similarity
- Response generation incorporating retrieved knowledge

### Avatar Synthesis

Avatars are synthesized using:
- Azure Speech SDK for text-to-speech
- Azure Talking Avatar service for facial animation
- WebRTC for real-time streaming to the browser

### Customization Options

- Multiple avatar characters (Dr. David Avenetti, Prof. Zalake, Lisa-Casual, Max-Business)
- Background selection with various themes
- Prompt customization with AI-assisted generation

## License

MIT License

## Acknowledgments

- Azure Cognitive Services Team
- Microsoft Azure OpenAI Service
- Pinecone Vector Database

---

This project demonstrates integration of multiple Azure services to create interactive, knowledge-grounded conversational avatars for various applications including education, customer service, and information delivery.


# How to activate (for developers at V-ARE Lab)

Due to high cost of involved models and services, we have paused/stopped certain necessary components on our Azure App Services. The following is how to activate all paused components:

1. Log in Micorosoft Azure home.
![Upload](https://github.com/V-ARE/AIAvatar/blob/d2cbac1c9d838ac550051618901911eb5009931d/static/image_2025-03-17_181905312.png)

2. Open the Web app 'VARELabUICivory' and start the website, if its paused. 
![Upload](https://github.com/V-ARE/AIAvatar/blob/d2cbac1c9d838ac550051618901911eb5009931d/static/image_2025-03-17_181956304.png)

3. Open the Azure cognitive serive 'secondTestFriday' and type in Pricing tier.
![Upload](https://github.com/V-ARE/AIAvatar/blob/d2cbac1c9d838ac550051618901911eb5009931d/static/image_2025-03-17_182053429.png)

4. In the pricing tier page, change the pricing to 'S0 Standard' and click apply.
![Upload](https://github.com/V-ARE/AIAvatar/blob/d2cbac1c9d838ac550051618901911eb5009931d/static/image_2025-03-17_182117958.png)


