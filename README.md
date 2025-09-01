
# GitaSphere AI
*Your Personal Guide to the Wisdom of the Bhagavad Gītā*


GitaSphere AI is a sophisticated, end-to-end AI application designed to make the profound teachings of the Bhagavad Gītā accessible to everyone. Whether you're a curious student or a dedicated scholar, GitaSphere provides instant translations and deep, structured interpretations of any verse, tailored to your level of understanding.

This project leverages a modern, multi-stage RAG (Retrieval-Augmented Generation) pipeline, powered by a fine-tuned, quantized Llama 3 model, to deliver accurate, context-aware, and citable answers, all while running efficiently on CPU infrastructure.
✨ Key Features

    Instant Sanskrit Translation: Automatically translates Devanagari script verses into English.

    Structured Four-Part Analysis: Every answer is broken down into:

        Importance: Why the verse is significant.

        Philosophical Interpretation: The core philosophical meaning.

        Modern Relevance: How the teaching applies to today's world.

        Practical Implementation: Actionable steps to apply the wisdom in your life.

    Multi-Persona Depth: Adjust the complexity of the explanation by choosing one of four personas: school, masters, phD, or bhakta.

    RAG for Accuracy: Utilizes a FAISS vector database of over 7,400 text chunks from authentic commentaries to ground the AI's responses and provide citable references, drastically reducing hallucinations.

    Efficient & Deployed: Fully containerized with Docker and deployed on AWS EC2, the application is highly efficient, with an average response time of under 50 seconds on CPU.

🚀 Live Application in Action

Here is a quick look at the user-friendly Gradio interface.
🛠️ Architecture & Tech Stack

The application follows a multi-stage pipeline to process user queries and generate high-quality responses.

Pipeline Flow:
User Input -> Gradio UI -> Sanskrit Translation -> RAG Search (FAISS) -> Prompt Engineering -> Llama 3 Inference -> Structured Output

Technology Stack:

Category
	
* *Technologies*
- AI / ML
- Llama-3 (Quantized GGUF), llama-cpp-python, Sentence Transformers, FAISS, RAG
* *Backend*

- Python, Gradio, FastAPI

* *Deployment*
* 
-Docker, AWS EC2 (m7i-flex.large), Nginx (Reverse Proxy), DDNS (No-IP)

Models on Hub
	
LLM Model, Knowledge Base
⚙️ How It Works

    Input & Translation: The Gradio interface captures the user's query. If the input is detected as Sanskrit, it is first translated into English using the googletrans library.

    Retrieval: The English query is encoded into a vector embedding. This vector is used to perform a similarity search against a pre-built FAISS index containing over 7,400 chunks from various Gītā commentaries.

    Reranking: The top results from the initial search are reranked using a more powerful Cross-Encoder model to improve relevance.

    Augmentation & Generation: The top-ranked, relevant text chunks are combined with the user's query and a carefully crafted system prompt (which includes the selected persona) and fed to the quantized Llama-3 model via llama-cpp-python.

    Output: The model generates a structured, four-part response, citing the retrieved context, which is then streamed back to the user interface.

📦 Getting Started & Local Setup

You can run the entire application locally using Docker.
Prerequisites

    Git

    Docker

Installation & Run

    Clone the repository:

    git clone <your-repo-link>
    cd gitasphere-ai

    Build the Docker image:
    This command builds the image using the provided Dockerfile, which handles all dependencies.

    docker build -t gitasphere-app .

    Run the container:
    This command starts the application. The models and knowledge base will be downloaded automatically from Hugging Face Hub on the first run.

    docker run -p 7860:7860 --name gitasphere-app gitasphere-app

    Access the application:
    Open your web browser and navigate to http://localhost:7860.

📜 License

This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.

<div align="center">
<em>Made with ❤️ and a passion for ancient wisdom.</em>
</div>

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
