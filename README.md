# RAG-Agent-with-LLM-NVIDIA
This repository contains a comprehensive implementation of a production-grade Retrieval-Augmented Generation (RAG) system developed using the NVIDIA AI stack. The project demonstrates the architecture, deployment, and evaluation of an intelligent agent capable of autonomous reasoning, structured data extraction, and semantic self-filtering.

🏗️ System Architecture & Microservices
The project is built on a containerized microservice architecture to ensure high availability and scalability.

Microservice Orchestration: Managed via Docker, the environment includes specialized services such as jupyter-notebook-server for development, a frontend chat interface, and an llm-service for model management.

NVIDIA NIM (NVIDIA Inference Microservices): The project utilizes NIMs optimized for scaled inference. These services support in-flight batching, allowing up to 256 active requests per node on DGX Cloud infrastructure.

Internal Routing: Services communicate via isolated network ports (e.g., port 9000 for LLM client/server interactions).

🧠 LLM Orchestration & Advanced Logic
Using LangChain Expression Language (LCEL), the project implements complex "building block" logic for agentic behavior.

Running State Chains: Implements a functional state management system where a "running state" dictionary propagates through the system. This allows the agent to maintain "Knowledge Base" variables that are updated dynamically as the conversation progresses.

Structured Data Extraction: Utilizing Pydantic models to define expected schemas for data like flight information or customer preferences. The system uses an RExtract module to force LLMs to fill these structured "slots" with validated data.

Production Deployment: Chains are served as REST API endpoints using LangServe and FastAPI, enabling standard communication with external frontends via RemoteRunnable.

📂 RAG Workflow & Document Engineering
The system is designed to ground LLM reasoning in verifiable external data sources.

Semantic Ingestion: Documents are loaded via specialized tools like ArxivLoader and split into semantically coherent chunks using RecursiveCharacterTextSplitter.

Vector Storage: High-performance semantic retrieval is managed through FAISS (Facebook AI Similarity Search), which stores document embeddings for fast similarity-based querying.

Contextual Optimization:

Bi-Encoders: Uses separate pathways for embed_query (shorter inputs) and embed_documents (long-form passages) to maximize mathematical alignment during retrieval.

LongContextReorder: Reorders retrieved documents to place the most relevant information at the beginning or end of the context window, solving the "lost in the middle" problem in LLM reasoning.

🛡️ Semantic Guardrailing
To ensure safety and relevance, the project implements a semantic guardrail mechanism.

Embedding Backbone: Instead of slow autoregressive LLM judging, the system uses high-speed embedding models (like nv-embed-v1) to vectorize inputs for classification.

Asynchronous Processing: Leverages Python’s asyncio to LODGE multiple embedding requests concurrently, dramatically reducing latency compared to serial processing.

Classifier Training: Includes routines to train shallow classifiers (Neural Networks or Logistic Regression) on top of frozen embedding backbones to predict the probability that a query is "Good" (relevant) or "Poor" (harmful/irrelevant).

📊 Evaluation & Metrics
The pipeline is numerically assessed using an LLM-as-a-Judge framework.

Synthetic Data Pairs: The system automatically samples document pools to generate "ground-truth" Question-Answer pairs.

Pairwise Evaluator: A judge LLM compares the RAG agent's response against the ground truth, outputting a numerical "Preference Score" and technical justification.

🛠️ Technical Stack
Models: Llama 3.1, Mixtral-8x22B, NVIDIA NV-Embed-v1.

Frameworks: LangChain, FastAPI, Gradio, Pydantic, Scikit-Learn.

Infrastructure: Docker, NVIDIA NIM, FAISS.

How to Run
Deploy Microservices: Use docker compose up -d within the composer/ directory to spin up the local inference and development environment.

Initialize Index: Run the document processing notebooks to generate a docstore_index/ using FAISS.

Launch Server: Run python server_app.py to expose the /basic_chat, /retriever, and /generator endpoints via LangServe.

Interface: Access the Gradio frontend via the reverse proxy link (default port 8090) to interact with the RAG agent.