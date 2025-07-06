---
layout: post
title:  "RAG Speedrun: Persistent Indexing with Polars and Parquet"
date:   2025-07-06 20:36:07 -0400
categories: python llm rag unstructured pdf parquet
---

This RAG Speedrun series has progressed through several stages. Initially, a basic Retrieval Augmented Generation system was developed, operating entirely in memory. Subsequently, the system was enhanced to process unstructured PDFs, recognizing that much valuable information resides within these document types.

A significant limitation of the previous approach was its performance. Each query necessitated re-reading all PDFs, re-extracting text, and re-computing embeddings. While acceptable for a small number of documents, this process becomes inefficient for larger datasets, leading to substantial delays. This approach is not conducive to rapid information retrieval.

Persistent indexing addresses this inefficiency by performing the computationally intensive tasks once, saving the results, and then loading them quickly for subsequent use. This concept is analogous to an organized library: books are cataloged and stored, allowing for efficient retrieval rather than re-creation each time they are needed.

For this endeavor, we introduce [Polars](https://pola.rs/) and [Parquet](https://parquet.apache.org/). Polars is a DataFrame library implemented in Rust, known for its high performance. In this context, its primary utility is its support for advanced data types, specifically the ability to store embedding vectors as arrays of floats within a column. While not as fast or memory-efficient as a memory-mapped NumPy array, the single-file nature of Parquet, which accommodates this Polars Array type, simplifies management compared to a two-file approach.

The updated script will feature two distinct commands: `index` and `query`. The `index` command will manage the parsing, embedding generation, and saving of data to a Parquet file. The `query` command will facilitate rapid loading of this pre-built index to answer queries efficiently.

For context, previous posts in this series include:
*   [RAG Speedrun: Local LLMs and Sentence Transformers](https://seantater.github.io/python/llm/rag/2025/07/04/rag-speedrun.html)
*   [RAG Speedrun: Local LLMs and Unstructured PDF Ingestion](https://seantater.github.io/python/llm/rag/unstructured/pdf/2025/07/05/rag-speedrun-unstructured-pdfs.html)

## Setting Up Your Environment

To replicate this setup, `uv` and Ollama are still required. Additionally, `polars` and `numpy` must be added to the Python environment.

1.  **UV**: This tool manages Python dependencies, specifically `polars` and `numpy` for this project, as indicated in the `uv` stanza at the top of the script.

2.  **Ollama**: This serves as the local LLM server. Ensure it is installed and a model such as `qwen3:4b` is pulled and running.
    *   Download and install Ollama: [https://ollama.com/](https://ollama.com/)
    *   Pull the model: `ollama run qwen3:4b`

## The RAG Speedrun Script with Persistent Indexing: Detailed Implementation

This section provides a detailed walkthrough of the Python script, `rag_persistent_index.py`, highlighting the components enabling persistent indexing.

### **Dependencies and Imports**

The script begins by specifying the required packages for `uv` and importing them. `polars` and `numpy` are new additions. Although direct NumPy array conversions for embeddings are minimized, Polars utilizes NumPy internally for its `to_torch()` method, necessitating its inclusion.

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai",
#     "sentence-transformers",
#     "torch",
#     "unstructured[pdf]", # Includes pdfminer.six, python-magic, lxml, etc
#     "polars", # For efficient DataFrame operations and Parquet I/O
#     "numpy", # For converting embeddings to numpy arrays (Polars uses it internally for torch interop)
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
import polars as pl
import uuid # For generating unique IDs
from datetime import datetime # For timestamps

# Ensure PyTorch uses CPU by default for broader compatibility
torch.set_default_device("cpu")

# Global model instance - we only want to load this once!
model = None

def load_sentence_transformer_model():
    """Loads the Sentence Transformer model if not already loaded."""
    global model
    if model is None:
        print("Loading Sentence Transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")
    return model
```
The `uuid` and `datetime` modules are introduced. While not strictly used in the application's core logic, their inclusion provides a more realistic representation of typical data management practices, where unique identifiers and timestamps are often crucial.

### **Enhanced Document Loading (`load_documents_from_pdf_directory`)**

The `load_documents_from_pdf_directory` function has been enhanced to attach essential metadata to each text chunk, beyond just raw text. This metadata is critical for tracking data provenance and its relationship to original documents.

```python
def load_documents_from_pdf_directory(pdf_dir):
    """
    Loads text content from all PDF files in a given directory and its subdirectories
    using unstructured, returning a list of dictionaries with document info.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.is_dir():
        raise ValueError(f"The provided path '{pdf_dir}' is not a valid directory.")

    all_document_chunks = []
    print(f"Searching for PDFs in '{pdf_dir}' recursively...")
    for pdf_file in pdf_path.rglob("*.pdf"):
        print(f"  Processing: {pdf_file}")
        doc_id = str(uuid.uuid4()) # Unique ID for the document
        chunk_number = 0
        try:
            elements = partition_pdf(filename=str(pdf_file), strategy="hi_res")
            for element in elements:
                if element.text:
                    all_document_chunks.append({
                        "filename": pdf_file.name,
                        "document_id": doc_id,
                        "chunk_number": chunk_number,
                        "text_snippet": element.text,
                        "index_timestamp": datetime.utcnow().isoformat(),
                        "embedding": None # Placeholder, will be filled later
                    })
                    chunk_number += 1
            if chunk_number == 0:
                print(f"    No text extracted from {pdf_file}")
        except Exception as e:
            print(f"    Error processing {pdf_file}: {e}")
    print(f"Finished loading {len(all_document_chunks)} text chunks from PDFs.")
    return all_document_chunks
```
The `doc_id`, `chunk_number`, and `index_timestamp` fields serve specific purposes:
*   **`document_id`**: A [UUID (Universally Unique Identifier)](https://en.wikipedia.org/wiki/Universally_unique_identifier) provides a highly unique identifier for each original PDF, ensuring distinct tracking even for files with identical names.
*   **`chunk_number`**: This field maintains the sequential order of text elements or chunks generated by `unstructured` within their original document. This is valuable for reconstructing documents or applying order-dependent logic.
*   **`index_timestamp`**: A UTC timestamp indicating when the chunk was processed, useful for debugging, version control, or data freshness assessment.

### **The `index` Command Implementation**

This section details the process of creating the persistent index. It involves loading the Sentence Transformer model, processing documents, generating embeddings, and preparing them for Polars and Parquet.

```python
def index_command(args):
    """
    Indexes PDF documents from a directory and saves embeddings to a Parquet file.
    """
    load_sentence_transformer_model()

    document_data = load_documents_from_pdf_directory(args.pdf_directory)
    if not document_data:
        print("No documents found or extracted for indexing. Exiting.")
        return

    print("Generating embeddings for document chunks...")
    texts = [d["text_snippet"] for d in document_data]
    embeddings = model.encode(texts, convert_to_tensor=False) # Get as numpy array

    # Assign embeddings back to the document_data list
    # We keep them as NumPy arrays for now, Polars can handle this directly
    for i, embedding in enumerate(embeddings):
        document_data[i]["embedding"] = embedding

    print("Creating Polars DataFrame...")
    df = pl.DataFrame(document_data)

    # Crucially, we cast the embedding column to a Polars Array type with a fixed dimension (384 for MiniLM-L6-v2)
    # and a specific float type (Float32). This enables direct conversion to a PyTorch tensor later.
    # If you don't do this, Polars will store it as a List, which cannot be directly converted to Torch.
    embedding_dimension = 384 # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    df = df.with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, embedding_dimension)))

    output_path = Path(args.output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    print(f"Saving index to {output_path}...")
    df.write_parquet(output_path)
    print("Indexing complete.")
```
A key aspect of this implementation is the handling of embeddings. After generation (as NumPy arrays), they are assigned to the `document_data` list. When creating the Polars DataFrame, a critical cast is performed: `pl.col("embedding").cast(pl.Array(pl.Float32, embedding_dimension))`.

This cast is essential because Polars offers two primary methods for storing sequences: `List` and `Array`.
*   A `List` column accommodates lists of varying lengths, offering flexibility but reduced performance for fixed-size vectors like embeddings. Importantly, `List` columns cannot be directly converted to a PyTorch tensor using Polars' `to_torch()` method.
*   An `Array` column is optimized for fixed-size arrays, such as 384-dimensional embeddings. This approach is more memory-efficient and, crucially, enables direct conversion of the entire column into a single, contiguous PyTorch tensor via the `to_torch()` method. Attempting to save as a `List` and then convert to a PyTorch tensor would typically involve multiple intermediate conversions (Python list of lists, then NumPy array), incurring significant memory copying and overhead, particularly for large datasets.

Finally, the structured Polars DataFrame is saved to a Parquet file, establishing the "persistent" aspect of the index. The decision to use Parquet, while not without its considerations, is made in the context of this tutorial to provide a practical, single-file solution.

### Reasons not to use Parquet
*   Parquet typically includes compression, which is often unnecessary for embeddings as they do not compress efficiently. This can be mitigated by disabling compression.
*   Parquet does not support `mmap`, a technique where the operating system maps a file directly into memory. This means Parquet will consume more memory than a memory-mapped NumPy array. However, if the index size exceeds available memory, performance degradation is inevitable regardless of the storage format.
*   Parquet files are immutable. Updating a document necessitates rewriting the entire index (copying unchanged portions) or implementing a layering technique where new files are written and all revisions are read during retrieval. Neither approach is optimal for frequently changing datasets, which is a primary reason to consider dedicated vector databases like Pinecone or Weaviate. In summary, large static datasets are less problematic than rapidly changing ones, and mutating existing documents poses more challenges than adding new ones. A dataset of one million chunks, with 1000 daily additions, can still be managed on a laptop. Query embedding time is independent of dataset size, and a million-chunk index can be queried in milliseconds, with further amortization for multiple concurrent queries.

### Reasons to use Parquet
*   Endorsing a specific vector database is outside the scope of this tutorial, as numerous offerings exist, each with its own documentation.
*   Using a separate SQLite database and embedding file for a local example would introduce complexities. While SQLite supports transactions, rolling back would leave the embedding file out of sync, compromising data integrity.
*   Employing separate JSON or other text files alongside an embedding file would increase complexity by requiring two distinct read operations, without offering significant speed or instructional benefits.

Consequently, Parquet is selected for this implementation.

### **Modified Retrieval Function (`retrieve_info`)**
 
The `retrieve_info` function has been refined to directly accept the PyTorch tensor from Polars, enhancing efficiency.

```python
def retrieve_info(query, document_df, document_embeddings_torch, top_k=2):
    """
    Retrieves the top_k most relevant document chunks for a given query
    from the loaded Polars DataFrame and PyTorch embeddings.
    """
    global model
    if model is None:
        load_sentence_transformer_model()

    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity between query and document embeddings
    # We can now directly use the PyTorch tensor from Polars! No more numpy intermediate.
    cosine_scores = util.cos_sim(query_embedding, document_embeddings_torch)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    # Retrieve the actual text snippets using the indices
    retrieved_chunks = [document_df["text_snippet"][idx] for idx in top_results.indices]
    return retrieved_chunks
```
The key improvement here is that `document_embeddings_torch` is already a PyTorch tensor, a result of the efficient conversion performed in the `query_command`. This eliminates the need for `torch.from_numpy()`, reducing data copying and accelerating operations.

### **The `query` Command Implementation**

This section describes the process of loading the pre-built index and preparing it for query answering, emphasizing speed.

```python
def query_command(args):
    """
    Loads an existing Parquet index and performs RAG queries.
    """
    input_path = Path(args.input_parquet_path)
    if not input_path.is_file():
        raise ValueError(f"The provided path '{input_path}' is not a valid Parquet file.")

    print(f"Loading index from {input_path}...")
    document_df = pl.read_parquet(input_path)
    print(f"Loaded {len(document_df)} document chunks.")

    # This is the magic! Convert the embedding column directly to a PyTorch tensor.
    # This is possible because we saved it as a Polars Array(Float32, 384).
    # This avoids the memory overhead of converting to a Python list or NumPy array first.
    document_embeddings_torch = document_df["embedding"].to_torch()
    print(f"Converted embeddings to PyTorch tensor with shape: {document_embeddings_torch.shape}")

    # Load the Sentence Transformer model once for queries
    load_sentence_transformer_model()

    # Process each query provided via CLI
    for query in args.queries:
        rag_pipeline(query, document_df, document_embeddings_torch)
```
The `pl.read_parquet(input_path)` command facilitates rapid index loading. The crucial step is `document_df["embedding"].to_torch()`. Because embeddings were meticulously saved as `pl.Array(pl.Float32, 384)` in the `index` command, Polars can directly and efficiently convert the entire column into a single PyTorch tensor. This significantly benefits memory usage and speed, particularly with large numbers of embeddings.

### **Main Execution Block (`if __name__ == "__main__":`)**

The main block now leverages `argparse` to establish a command-line interface with `index` and `query` subcommands, enhancing script usability and organization.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Speedrun with Persistent Indexing using Polars and Parquet.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Index command parser
    index_parser = subparsers.add_parser("index", help="Index PDF documents and save embeddings to a Parquet file.")
    index_parser.add_argument("pdf_directory", type=str,
                              help="Path to the directory containing PDF documents.")
    index_parser.add_argument("output_parquet_path", type=str,
                              help="Path to the output Parquet file for the index.")
    index_parser.set_defaults(func=index_command)

    # Query command parser
    query_parser = subparsers.add_parser("query", help="Load an existing Parquet index and perform RAG queries.")
    query_parser.add_argument("input_parquet_path", type=str,
                              help="Path to the input Parquet file containing the index.")
    query_parser.add_argument("queries", type=str, nargs='+',
                              help="One or more queries to ask the RAG system.")
    query_parser.set_defaults(func=query_command)

    args = parser.parse_args()
    args.func(args)
```

### Full RAG Speedrun Script with Persistent Indexing

For reference, the complete Python script is provided below.

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "openai",
#     "sentence-transformers",
#     "torch",
#     "unstructured[pdf]", # Includes pdfminer.six, python-magic, lxml, etc
#     "polars[pyarrow]", # For efficient DataFrame operations and Parquet I/O
#     "numpy", # For converting embeddings to numpy arrays (Polars uses it internally for torch interop)
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
import polars as pl
import numpy as np
import uuid
from datetime import datetime

# Ensure PyTorch uses CPU by default for broader compatibility
torch.set_default_device("cpu")

# Global model instance
model = None

def load_sentence_transformer_model():
    """Loads the Sentence Transformer model if not already loaded."""
    global model
    if model is None:
        print("Loading Sentence Transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")
    return model

def load_documents_from_pdf_directory(pdf_dir):
    """
    Loads text content from all PDF files in a given directory and its subdirectories
    using unstructured, returning a list of dictionaries with document info.
    """
    pdf_path = Path(pdf_dir)
    if not pdf_path.is_dir():
        raise ValueError(f"The provided path '{pdf_dir}' is not a valid directory.")

    all_document_chunks = []
    print(f"Searching for PDFs in '{pdf_dir}' recursively...")
    for pdf_file in pdf_path.rglob("*.pdf"):
        print(f"  Processing: {pdf_file}")
        doc_id = str(uuid.uuid4()) # Unique ID for the document
        chunk_number = 0
        try:
            elements = partition_pdf(filename=str(pdf_file), strategy="hi_res")
            for element in elements:
                if element.text:
                    all_document_chunks.append({
                        "filename": pdf_file.name,
                        "document_id": doc_id,
                        "chunk_number": chunk_number,
                        "text_snippet": element.text,
                        "index_timestamp": datetime.utcnow().isoformat(),
                        "embedding": None # Placeholder, will be filled later
                    })
                    chunk_number += 1
            if chunk_number == 0:
                print(f"    No text extracted from {pdf_file}")
        except Exception as e:
            print(f"    Error processing {pdf_file}: {e}")
    print(f"Finished loading {len(all_document_chunks)} text chunks from PDFs.")
    return all_document_chunks

def index_command(args):
    """
    Indexes PDF documents from a directory and saves embeddings to a Parquet file.
    """
    load_sentence_transformer_model()

    document_data = load_documents_from_pdf_directory(args.pdf_directory)
    if not document_data:
        print("No documents found or extracted for indexing. Exiting.")
        return

    print("Generating embeddings for document chunks...")
    texts = [d["text_snippet"] for d in document_data]
    embeddings = model.encode(texts, convert_to_tensor=False) # Get as numpy array

    # Assign embeddings back to the document_data list
    # We keep them as NumPy arrays for now, Polars can handle this directly
    for i, embedding in enumerate(embeddings):
        document_data[i]["embedding"] = embedding

    print("Creating Polars DataFrame...")
    df = pl.DataFrame(document_data)

    # Crucially, we cast the embedding column to a Polars Array type with a fixed dimension (384 for MiniLM-L6-v2)
    # and a specific float type (Float32). This enables direct conversion to a PyTorch tensor later.
    # If you don't do this, Polars will store it as a List, which cannot be directly converted to Torch.
    embedding_dimension = 384 # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    df = df.with_columns(pl.col("embedding").cast(pl.Array(pl.Float32, embedding_dimension)))

    output_path = Path(args.output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    print(f"Saving index to {output_path}...")
    df.write_parquet(output_path)
    print("Indexing complete.")

def retrieve_info(query, document_df, document_embeddings_torch, top_k=2):
    """
    Retrieves the top_k most relevant document chunks for a given query
    from the loaded Polars DataFrame and PyTorch embeddings.
    """
    global model
    if model is None:
        load_sentence_transformer_model()

    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity between query and document embeddings
    # We can now directly use the PyTorch tensor from Polars! No more numpy intermediate.
    cosine_scores = util.cos_sim(query_embedding, document_embeddings_torch)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    # Retrieve the actual text snippets using the indices
    retrieved_chunks = [document_df["text_snippet"][idx] for idx in top_results.indices]
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

def rag_pipeline(query, document_df, document_embeddings_torch):
    """
    Orchestrates the RAG pipeline: retrieve and then generate.
    """
    print(f"\n--- Processing Query: '{query}' ---")
    retrieved_chunks = retrieve_info(query, document_df, document_embeddings_torch)
    print("\nRetrieved Context:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  {i+1}. {chunk}")

    context_str = "\n".join(retrieved_chunks)
    response = generate_response_with_ollama(query, context_str)
    print("\nGenerated Response:")
    print(response)
    print("------------------------------------")

def query_command(args):
    """
    Loads an existing Parquet index and performs RAG queries.
    """
    input_path = Path(args.input_parquet_path)
    if not input_path.is_file():
        raise ValueError(f"The provided path '{input_path}' is not a valid Parquet file.")

    print(f"Loading index from {input_path}...")
    document_df = pl.read_parquet(input_path)
    print(f"Loaded {len(document_df)} document chunks.")

    # This is the magic! Convert the embedding column directly to a PyTorch tensor.
    # This is possible because we saved it as a Polars Array(Float32, 384).
    # This avoids the memory overhead of converting to a Python list or NumPy array first.
    document_embeddings_torch = document_df["embedding"].to_torch()
    print(f"Converted embeddings to PyTorch tensor with shape: {document_embeddings_torch.shape}")

    # Load the Sentence Transformer model once for queries
    load_sentence_transformer_model()

    # Process each query provided via CLI
    for query in args.queries:
        rag_pipeline(query, document_df, document_embeddings_torch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Speedrun with Persistent Indexing using Polars and Parquet.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Index command parser
    index_parser = subparsers.add_parser("index", help="Index PDF documents and save embeddings to a Parquet file.")
    index_parser.add_argument("pdf_directory", type=str,
                              help="Path to the directory containing PDF documents.")
    index_parser.add_argument("output_parquet_path", type=str,
                              help="Path to the output Parquet file for the index.")
    index_parser.set_defaults(func=index_command)

    # Query command parser
    query_parser = subparsers.add_parser("query", help="Load an existing Parquet index and perform RAG queries.")
    query_parser.add_argument("input_parquet_path", type=str,
                              help="Path to the input Parquet file containing the index.")
    query_parser.add_argument("queries", type=str, nargs='+',
                              help="One or more queries to ask the RAG system.")
    query_parser.set_defaults(func=query_command)

    args = parser.parse_args()
    args.func(args)
```