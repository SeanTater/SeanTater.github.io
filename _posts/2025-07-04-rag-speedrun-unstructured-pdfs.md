---
layout: post
title:  "RAG Speedrun: Local LLMs and Unstructured PDF Ingestion"
date:   2025-07-04 21:36:07 -0400
categories: python llm rag unstructured pdf
---

Building on our previous RAG Speedrun, where we demonstrated a basic Retrieval Augmented Generation system with in-memory documents, this post will tackle a common real-world challenge: ingesting information from unstructured PDF documents. In many enterprise and research settings, valuable knowledge is locked away in PDFs, making it difficult for LLMs to access.

This is where `unstructured` comes in. `unstructured` is a powerful library designed to extract clean, structured content from a variety of document types, including PDFs, HTML, Word documents, and more. It handles the complexities of parsing different layouts, tables, and text elements, providing a unified output that's ready for further processing, like generating embeddings for RAG.

In this "RAG Speedrun" extension, we'll modify our existing RAG system to read documents from a local directory of PDFs. We'll use `unstructured` to parse these PDFs and `Pathlib` with `glob` to recursively find all PDF files within a specified folder, which the user will provide as a command-line argument. The goal is to show how easily you can integrate real-world document ingestion into your RAG pipeline.

## The Challenge of Unstructured Data

PDFs are notorious for being difficult to parse programmatically. They often contain complex layouts, images, tables, and various text encodings. Simply extracting raw text can lead to a jumbled mess that's unusable for RAG. `unstructured` addresses this by intelligently identifying and categorizing elements within the document, providing a more coherent and semantically meaningful output.

## Setting Up Your Environment

To follow along, you'll need `uv` and Ollama, as in the previous post. Additionally, we'll add `unstructured` and its PDF-specific dependencies.

1.  **UV** will install your Python dependencies. We'll be adding **`unstructured`**, a library for document parsing, often used in machine learning apps.

    The script below includes an updated `uv` stanza to automatically install these new dependencies.

2.  **Ollama**: For running a local Large Language Model. Ensure you have Ollama installed and a model like `qwen3:4b` pulled and running.
    *   Download and install Ollama: [https://ollama.com/](https://ollama.com/)
    *   Pull the model: `ollama run qwen3:4b`

## The RAG Speedrun Script with PDF Ingestion

Let's walk through the updated Python script. We'll focus on the changes required to incorporate PDF ingestion using `unstructured`.

### Imports and Argument Parsing

First, we add `pathlib` for file system operations and `argparse` to accept the PDF directory and one or more queries as command-line arguments. We also add `unstructured.partition.pdf` for PDF parsing.

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai",
#     "sentence-transformers",
#     "torch",
#     "unstructured[pdf]", # Includes pdfminer.six, python-magic, lxml
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
import argparse
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Ensure PyTorch uses CPU by default for broader compatibility
torch.set_default_device("cpu")
```

### Document Loading from PDFs

Instead of a hardcoded `documents` list, we'll now implement a function to read PDFs from a specified directory. This function will use `Pathlib` to find all `.pdf` files recursively and `unstructured.partition_pdf` to extract text.

```python
def load_documents_from_pdf_directory(pdf_dir):
    """
    Loads text content from all PDF files in a given directory and its subdirectories
    using unstructured.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.is_dir():
        raise ValueError(f"The provided path '{pdf_dir}' is not a valid directory.")

    all_documents = []
    print(f"Searching for PDFs in '{pdf_dir}' recursively...")
    for pdf_file in pdf_path.rglob("*.pdf"):
        print(f"  Processing: {pdf_file}")
        try:
            # Use partition_pdf to extract elements from the PDF
            elements = partition_pdf(filename=str(pdf_file), strategy="hi_res")
            # Join the text from all elements to form a single document string
            document_text = "\n\n".join([str(el) for el in elements if el.text])
            if document_text:
                all_documents.append(document_text)
            else:
                print(f"    No text extracted from {pdf_file}")
        except Exception as e:
            print(f"    Error processing {pdf_file}: {e}")
    print(f"Finished loading {len(all_documents)} documents.")
    return all_documents
```

### Main Execution Block (`if __name__ == "__main__":`)

The main block will now parse the command-line arguments for the PDF directory and queries, load the documents, and then proceed with the RAG pipeline for each query.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Speedrun with PDF ingestion using Unstructured.")
    parser.add_argument("pdf_directory", type=str,
                        help="Path to the directory containing PDF documents.")
    parser.add_argument("queries", type=str, nargs='+',
                        help="One or more queries to ask the RAG system.")
    args = parser.parse_args()

    # 1. Load documents from the specified PDF directory
    documents = load_documents_from_pdf_directory(args.pdf_directory)

    if not documents:
        print("No documents found or extracted. Exiting.")
        exit()

    # 2. Load a pre-trained Sentence Transformer model
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")

    # 3. Generate embeddings for the documents
    print("Generating document embeddings...")
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    print("Document embeddings generated.")

    # Process each query provided via CLI
    for query in args.queries:
        rag_pipeline(query)
```

The `retrieve_info`, `generate_response_with_ollama`, and `rag_pipeline` functions remain largely the same, as their core logic is independent of how the documents are loaded.

### Full RAG Speedrun Script with Unstructured PDF Ingestion

Here is the complete Python script. Save it as `rag_unstructured_pdfs.py` (or any other `.py` file).

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai",
#     "sentence-transformers",
#     "torch",
#     "unstructured[pdf]", # Includes pdfminer.six, python-magic, lxml, etc
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
import argparse
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Ensure PyTorch uses CPU by default for broader compatibility
torch.set_default_device("cpu")

def load_documents_from_pdf_directory(pdf_dir):
    """
    Loads text content from all PDF files in a given directory and its subdirectories
    using unstructured.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.is_dir():
        raise ValueError(f"The provided path '{pdf_dir}' is not a valid directory.")

    all_documents = []
    print(f"Searching for PDFs in '{pdf_dir}' recursively...")
    for pdf_file in pdf_path.rglob("*.pdf"):
        print(f"  Processing: {pdf_file}")
        try:
            # Use partition_pdf to extract elements from the PDF
            elements = partition_pdf(filename=str(pdf_file), strategy="hi_res")
            # Join the text from all elements to form a single document string
            document_text = "\n\n".join([str(el) for el in elements if el.text])
            if document_text:
                all_documents.append(document_text)
            else:
                print(f"    No text extracted from {pdf_file}")
        except Exception as e:
            print(f"    Error processing {pdf_file}: {e}")
    print(f"Finished loading {len(all_documents)} documents.")
    return all_documents

# 2. Load a pre-trained Sentence Transformer model (moved outside main for clarity, but initialized once)
# 'all-MiniLM-L6-v2' is a good balance of performance and size.
# This will be loaded once the script starts, after documents are loaded.
model = None # Will be initialized in main

def retrieve_info(query, top_k=2):
    """
    Retrieves the top_k most relevant document chunks for a given query.
    """
    if model is None:
        raise RuntimeError("SentenceTransformer model not loaded. Call model = SentenceTransformer(...) first.")
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
    parser = argparse.ArgumentParser(description="RAG Speedrun with PDF ingestion using Unstructured.")
    parser.add_argument("pdf_directory", type=str,
                        help="Path to the directory containing PDF documents.")
    parser.add_argument("queries", type=str, nargs='+',
                        help="One or more queries to ask the RAG system.")
    args = parser.parse_args()

    # 1. Load documents from the specified PDF directory
    documents = load_documents_from_pdf_directory(args.pdf_directory)

    if not documents:
        print("No documents found or extracted. Exiting.")
        exit()

    # 2. Load a pre-trained Sentence Transformer model
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")

    # 3. Generate embeddings for the documents
    print("Generating document embeddings...")
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    print("Document embeddings generated.")

    # Process each query provided via CLI
    for query in args.queries:
        rag_pipeline(query)
```

## How to Run

1.  Save the Python code above as `rag_unstructured_pdfs.py` in your project directory.
2.  Ensure Ollama is running and you have the `qwen3:4b` model pulled (`ollama run qwen3:4b`).
3.  Create a directory (e.g., `my_pdfs`) and place some PDF files inside it. For testing, you can create simple PDFs with the facts from the previous blog post (e.g., one PDF stating "The company's annual 'Innovation Summit' is held every October in the virtual metaverse.", another stating "Our new employee onboarding process requires completion of the 'Clarity Protocol' module within the first week.", etc.).
4.  Run the script from your terminal, providing the path to your PDF directory and your queries:

    ```bash
    uv run rag_unstructured_pdfs.py my_pdfs/ "When is the Innovation Summit held?" "What is the Clarity Protocol?"
    ```
    Replace `my_pdfs/` with the actual path to your directory containing PDFs, and provide your desired queries.

## What's next

If you actually ran the script above (rather than just reading the post like we all know you did!), you'd notice this system is slow as molasses. In my case, it takes nearly thirty minutes to build the index each time. It parses all the PDFs, with OCR, on every run. This is obviously not how you would want to do this in a production environment, or even just for a second run. We'll cover that, with about the simplest imaginable implementation, in the next post.


