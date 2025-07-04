---
layout: post
title:  "RAG Speedrun: Local LLMs and Sentence Transformers"
date:   2025-07-03 20:36:07 -0400
categories: python llm rag
---

Being a data scientist by trade, one of the first things I want to do with any new technology is get it working quickly. You've probably heard the buzz around Large Language Models (LLMs) and their incredible ability to generate human-like text. However, LLMs often struggle with providing up-to-date or domain-specific information, as their knowledge is limited to their training data. This is where Retrieval Augmented Generation (RAG) comes in.

RAG is a powerful technique that combines the strengths of information retrieval systems with the generative capabilities of LLMs. Instead of relying solely on the LLM's internal knowledge, RAG first retrieves relevant information from an external knowledge base and then uses that information to guide the LLM's response. This allows LLMs to provide more accurate, current, and contextually relevant answers, reducing hallucinations and grounding their responses in factual data.

In this "RAG Speedrun," we'll build a simple, yet functional, RAG system using just a single Python script. We'll leverage `sentence-transformers` for efficient text embeddings and a local LLM (powered by Ollama) for text generation. The goal is to get you from zero to a working RAG system in minutes, demonstrating the core concepts without getting bogged down in complex infrastructure.

## The Core Components of RAG

At its heart, a RAG system involves three main stages:

*   **Embeddings**: This is the process of converting text into numerical vectors (embeddings) that capture their semantic meaning. Texts with similar meanings will have embeddings that are close to each other in a multi-dimensional space. We'll use `sentence-transformers` for this, which provides pre-trained models optimized for generating high-quality sentence and paragraph embeddings.
*   **Retrieval**: Once our knowledge base (a collection of documents or text chunks) is embedded, when a user asks a question, we convert their query into an embedding. We then compare this query embedding to all the document embeddings to find the most semantically similar chunks. These are the "relevant" pieces of information that the LLM will use.
*   **Generation**: Finally, the retrieved relevant text chunks are passed to a Large Language Model along with the original user query. The LLM then uses this augmented context to generate a coherent and informed response. By providing the LLM with specific, relevant information, we guide its output and ensure it stays on topic and factual.

## Setting Up Your Environment

To follow along with this speedrun, you'll need Python installed. We'll use two main dependencies:

1.  **`sentence-transformers`**: For generating text embeddings.
    ```bash
    pip install sentence-transformers openai
    ```
2.  **Ollama**: For running a local Large Language Model. Ollama makes it incredibly easy to download and run various open-source LLMs on your machine.
    *   Download and install Ollama from their official website: [https://ollama.com/](https://ollama.com/)
    *   Once installed, you can download a model. For this example, we'll use `qwen3:4b`, which has shown good performance for these types of questions:
        ```bash
        ollama run qwen3:4b
        ```
        This command will download `qwen3:4b` if you don't have it and start it. Keep this running in a separate terminal, or you can interact with it via its API. For our Python script, we'll interact with its local API.

    *   **UV (Python Package Installer)**: While `pip` is common, `uv` is a new, fast Python package installer and resolver. If you don't have it, you can install it:
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
        The script below includes a stanza at the top to automatically install dependencies using `uv`.

## The RAG Speedrun Script

Let's walk through the single Python script that brings our RAG system to life. We'll break it down into logical parts, explaining each segment, and then provide the full, runnable code at the end.

### Imports and Document Corpus

First, we import the necessary libraries and define our small, in-memory knowledge base. We've chosen distinct, made-up facts to clearly demonstrate how the embedding and retrieval process helps in finding relevant information that an LLM wouldn't know by default.

The script begins with a special `uv` stanza, similar to a `pyproject.toml` file, which declares its dependencies. This allows `uv` to automatically install the required packages when the script is run.

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai",
#     "sentence-transformers",
#     "torch",
# ]
#
# # The following section uses the CPU version of pytorch by default, since it is smaller and more portable,
# # but you can remove the lines below to use the Nvidia GPU version if you have a compatible GPU.
#
# [tool.uv.sources]
# torch = { index = "pytorch" }
#
# [[tool.uv.index]]
# name = "pytorch"
# url = "https://download.pytorch.org/whl/cpu"
# ///
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Ensure PyTorch uses CPU by default for broader compatibility
torch.set_default_device("cpu")

# 1. Define a small, in-memory knowledge base
# We'll use distinct, made-up facts to demonstrate RAG's ability to retrieve non-public knowledge.
documents = [
    "The company's annual 'Innovation Summit' is held every October in the virtual metaverse.",
    "Our new employee onboarding process requires completion of the 'Clarity Protocol' module within the first week.",
    "Project Nightingale's primary objective is to integrate AI-driven analytics into legacy systems by Q3.",
    "The best coffee machine in the office is located on the 7th floor, near the quantum computing lab.",
    "Employee benefits include unlimited access to the 'Mindfulness Pods' located on floors 3 and 5.",
    "The internal code review guidelines emphasize readability and a maximum of 80 characters per line.",
    "Our next team-building event will be a virtual escape room challenge on the last Friday of next month.",
    "The 'Quantum Leap' initiative aims to reduce computational overhead by 40% by the end of the fiscal year.",
    "For expense reports, all receipts must be submitted via the 'Nexus Portal' within 48 hours of the transaction.",
]
```

### Model Loading and Embedding Generation

Next, we load our `SentenceTransformer` model. `all-MiniLM-L6-v2` is a great choice for its balance of performance and size, making it suitable for quick experiments. We then generate embeddings for all our `documents`. This is a one-time process for our static knowledge base.

```python
# 2. Load a pre-trained Sentence Transformer model
# 'all-MiniLM-L6-v2' is a good balance of performance and size.
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# 3. Generate embeddings for the documents
print("Generating document embeddings...")
document_embeddings = model.encode(documents, convert_to_tensor=True)
print("Document embeddings generated.")
```

### Retrieval Function (`retrieve_info`)

This function takes a user query, converts it into an embedding, and then calculates the cosine similarity against all our pre-computed document embeddings. Cosine similarity measures the angle between two vectors, indicating how similar their directions are. A higher score means greater semantic similarity. We then return the `top_k` most relevant document chunks.

```python
def retrieve_info(query, top_k=2):
    """
    Retrieves the top_k most relevant document chunks for a given query.
    """
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity between query and document embeddings
    # and get the top_k results.
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    retrieved_chunks = [documents[idx] for idx in top_results.indices]
    return retrieved_chunks
```

### Generation Function (`generate_response_with_ollama`)

This function handles the interaction with our local LLM running via Ollama. We use the `openai` Python client, pointing it to Ollama's local API endpoint. We construct a prompt that includes both the user's question and the retrieved context, instructing the LLM to answer based *only* on the provided context. This is crucial for grounding the LLM's response.

```python
def generate_response_with_ollama(query, context):
    """
    Generates a response using a local Ollama LLM, augmented with context,
    via the OpenAI-compatible API.
    """
    client = OpenAI(
        base_url="http://localhost:11434/v1", # Default Ollama API endpoint for OpenAI compatibility
        api_key="ollama", # Required, but can be any string
    )
    model_name = "qwen3:4b" # Ensure you have this model pulled with 'ollama run qwen3:4b'

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Based on the provided context, answer the question. If the answer is not in the context, state that you don't have enough information."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    print(f"\nSending request to Ollama with model: {model_name}...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during Ollama API call: {e}\nEnsure Ollama is running and the '{model_name}' model is pulled (run 'ollama run {model_name}' in your terminal)."
```

### RAG Pipeline Orchestration (`rag_pipeline`)

This function ties everything together. It takes a query, calls `retrieve_info` to get relevant chunks, formats these chunks into a single context string, and then passes this context along with the query to `generate_response_with_ollama`. It also includes print statements to show the retrieved context and the final generated response.

```python
def rag_pipeline(query):
    """
    Orchestrates the RAG pipeline: retrieve and then generate.
    """
    print(f"\n--- Processing Query: '{query}' ---")
    retrieved_chunks = retrieve_info(query)
    print("\nRetrieved Context:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  {i+1}. {chunk}")

    context_str = "\n".join(retrieved_chunks)
    response = generate_response_with_ollama(query, context_str)
    print("\nGenerated Response:")
    print(response)
    print("------------------------------------")
```

### Running the Pipeline

Finally, the `if __name__ == "__main__":` block demonstrates how to use our `rag_pipeline` with a few example queries. Notice the last query ("What is the best color?") is intentionally outside our knowledge base to show how the LLM will respond when it lacks sufficient context.

```python
if __name__ == "__main__":
    # Example queries
    rag_pipeline("When is the Innovation Summit held?")
    rag_pipeline("What is the Clarity Protocol?")
    rag_pipeline("Where can I find the best coffee machine?")
    rag_pipeline("What is the company's favorite animal?") # Query outside the context
```

### Full RAG Speedrun Script

For your convenience, here is the complete Python script that you can copy and paste to run your own RAG speedrun. Note the `/// script` stanza at the top, which is similar to a `pyproject.toml` file and allows `uv` to automatically manage dependencies.

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai",
#     "sentence-transformers",
#     "torch",
# ]
#
# # The following section uses the CPU version of pytorch by default, since it is smaller and more portable,
# # but you can remove the lines below to use the Nvidia GPU version if you have a compatible GPU.
#
# [tool.uv.sources]
# torch = { index = "pytorch" }
#
# [[tool.uv.index]]
# name = "pytorch"
# url = "https://download.pytorch.org/whl/cpu"
# ///
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Ensure PyTorch uses CPU by default for broader compatibility
torch.set_default_device("cpu")

# 1. Define a small, in-memory knowledge base
# We'll use distinct, made-up facts to demonstrate RAG's ability to retrieve non-public knowledge.
documents = [
    "The company's annual 'Innovation Summit' is held every October in the virtual metaverse.",
    "Our new employee onboarding process requires completion of the 'Clarity Protocol' module within the first week.",
    "Project Nightingale's primary objective is to integrate AI-driven analytics into legacy systems by Q3.",
    "The best coffee machine in the office is located on the 7th floor, near the quantum computing lab.",
    "Employee benefits include unlimited access to the 'Mindfulness Pods' located on floors 3 and 5.",
    "The internal code review guidelines emphasize readability and a maximum of 80 characters per line.",
    "Our next team-building event will be a virtual escape room challenge on the last Friday of next month.",
    "The 'Quantum Leap' initiative aims to reduce computational overhead by 40% by the end of the fiscal year.",
    "For expense reports, all receipts must be submitted via the 'Nexus Portal' within 48 hours of the transaction.",
]

# 2. Load a pre-trained Sentence Transformer model
# 'all-MiniLM-L6-v2' is a good balance of performance and size.
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# 3. Generate embeddings for the documents
print("Generating document embeddings...")
document_embeddings = model.encode(documents, convert_to_tensor=True)
print("Document embeddings generated.")

def retrieve_info(query, top_k=2):
    """
    Retrieves the top_k most relevant document chunks for a given query.
    """
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity between query and document embeddings
    # and get the top_k results.
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    retrieved_chunks = [documents[idx] for idx in top_results.indices]
    return retrieved_chunks

def generate_response_with_ollama(query, context):
    """
    Generates a response using a local Ollama LLM, augmented with context,
    via the OpenAI-compatible API.
    """
    client = OpenAI(
        base_url="http://localhost:11434/v1", # Default Ollama API endpoint for OpenAI compatibility
        api_key="ollama", # Required, but can be any string
    )
    model_name = "qwen3:4b" # Ensure you have this model pulled with 'ollama run qwen3:4b'

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Based on the provided context, answer the question. If the answer is not in the context, state that you don't have enough information."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    print(f"\nSending request to Ollama with model: {model_name}...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during Ollama API call: {e}\nEnsure Ollama is running and the '{model_name}' model is pulled (run 'ollama run {model_name}' in your terminal)."

def rag_pipeline(query):
    """
    Orchestrates the RAG pipeline: retrieve and then generate.
    """
    print(f"\n--- Processing Query: '{query}' ---")
    retrieved_chunks = retrieve_info(query)
    print("\nRetrieved Context:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  {i+1}. {chunk}")

    context_str = "\n".join(retrieved_chunks)
    response = generate_response_with_ollama(query, context_str)
    print("\nGenerated Response:")
    print(response)
    print("------------------------------------")

if __name__ == "__main__":
    # Example queries
    rag_pipeline("When is the Innovation Summit held?")
    rag_pipeline("What is the Clarity Protocol?")
    rag_pipeline("Where can I find the best coffee machine?")
    rag_pipeline("What is the company's favorite animal?") # Query outside the context
```

## Conclusion

Congratulations! You've just built a functional Retrieval Augmented Generation system from scratch with a single Python script. This speedrun demonstrates the core principles of RAG: using embeddings for semantic search to retrieve relevant context and then leveraging a local LLM to generate informed responses.

This simple setup is powerful because it allows you to:
*   **Ground LLM responses**: Reduce hallucinations by providing factual, external information.
*   **Incorporate up-to-date knowledge**: Easily update your knowledge base without retraining the LLM.
*   **Handle domain-specific queries**: Provide answers tailored to your specific data.

While this is a basic example, it lays the foundation for more complex RAG applications. You can expand upon this by:
*   Using larger, more diverse knowledge bases (e.g., loading from files, databases).
*   Implementing more sophisticated retrieval methods (e.g., hybrid search, re-ranking).
*   Experimenting with different `sentence-transformers` models or local LLMs.
