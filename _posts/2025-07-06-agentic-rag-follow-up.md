---
layout: post
title: "Agentic RAG: Beyond Simple Query-Response"
date: 2025-07-06 11:15:30 -0400
categories: [python, llm, rag, agentic]
---

In a previous post, "[RAG Speedrun: From Zero to Retrieval-Augmented Generation in Python](https://seantater.github.io/python/llm/rag/2025/07/04/rag-speedrun.html)", we introduced the core concepts of Retrieval-Augmented Generation (RAG). We demonstrated how to combine an external knowledge base with a Large Language Model (LLM) to enable it to answer questions that go beyond its initial training data. The fundamental approach involved a two-step process: first, retrieving relevant information based on a user's query, and then incorporating that information directly into the LLM's prompt for generating a response.

This "query-first, then generate" paradigm, while effective for many use cases, operates under certain assumptions. It presumes that the initial query is perfectly formulated for retrieval and that a single retrieval step is sufficient to gather all necessary context. However, real-world scenarios are often more complex. What if the LLM needs to perform multiple, iterative searches? What if it needs to refine its search queries based on initial results, or even decide autonomously when it has accumulated enough information to confidently answer the user's question?

This is precisely where the power of **Agentic RAG** becomes apparent. Agentic RAG represents an evolution in how LLMs interact with external data sources. Instead of being passive recipients of pre-retrieved information, LLMs are empowered with the ability to act as intelligent agents. They can dynamically interact with their environment by calling specific functions or "tools" as needed.

In an Agentic RAG system, the LLM is no longer confined to a rigid, predefined sequence of operations. It transforms into an active participant that can decide *when* to retrieve information, *how* to formulate its retrieval queries, and ultimately, *when* to synthesize a final, comprehensive response. This post will delve into the concept, implementation details, and significant benefits of building such an Agentic RAG system.

## The Agentic RAG Plan

Our Agentic RAG system will operate as a loop, allowing the LLM to dynamically decide its next action. The core idea is to present the LLM with the user's query and a set of available functions (tools) it can call. The LLM will then choose to:

1.  **Perform a search**: If it needs more information, it will call a `search` function, providing a query for the search. The results of this search will be fed back to the LLM.
2.  **Complete the response**: If it believes it has sufficient information, it will call a `complete_response` function, providing its final answer.

This loop will continue for a maximum of 5 iterations to prevent infinite loops, failing if a final response isn't generated within this limit.

Here's a high-level flow:

<pre class="mermaid">
graph TD
    A[Start: User Query] --> B{Initialize Conversation};
    B --> C{Loop up to 5 times};
    C --> D[LLM decides action];
    D -- Calls search tool --> E[Execute Search Function];
    E --> F[Add Search Results to Conversation];
    F --> C;
    D -- Calls complete_response tool --> G[Execute Complete Response Function];
    G --> H[Return Final Answer];
    H --> I[End];
    C -- Max iterations reached / No tool call --> J[Fail: No final answer];
    J --> I;
</pre>

## Code Segments

We'll build upon the `speedrun.py` script, reusing the document embeddings and the `retrieve_info` function. The main changes will be in how we interact with the LLM and orchestrate the RAG pipeline.

### 1. Setting up the Tools for the LLM

For the LLM to "know" about the functions it can call, we need to define them in a structured format that the OpenAI-compatible API understands. This is done by providing a list of dictionaries, where each dictionary describes a function, its purpose, and its parameters.

```python
# Define tools for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Searches the knowledge base for relevant information based on a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_response",
            "description": "Provides the final answer to the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete answer to the user's question.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]
```

### 2. The Agentic RAG Pipeline: Orchestrating LLM and Tools

This is the core of our agent. The `agentic_rag_pipeline` function manages the entire interaction loop: it sends messages to the LLM, interprets the LLM's responses (whether it's a direct answer or a tool call), executes the requested tools, and feeds the tool's output back into the conversation.

```python
def agentic_rag_pipeline(user_query, max_iterations=5):
    """
    Orchestrates the Agentic RAG pipeline, allowing the LLM to use tools.
    This function manages the conversation flow, calls the LLM, and executes
    tools based on the LLM's decisions.
    """
    print(f"\n--- Processing Agentic Query: '{user_query}' ---")
    # Initialize the OpenAI client to interact with the Ollama LLM
    client = OpenAI(
        base_url="http://localhost:11434/v1", # Default Ollama API endpoint
        api_key="ollama", # Required, but can be any string for Ollama
    )
    model_name = "qwen3:4b" # Specify the model to use (ensure it's pulled in Ollama)

    # Initialize the conversation with a system message and the user's query.
    # The system message instructs the LLM on its capabilities and how to use the tools.
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You can search a knowledge base for information and provide a final answer. Use the 'search' tool if you need more information. Use the 'complete_response' tool when you have a complete answer to the user's question."},
        {"role": "user", "content": user_query}
    ]

    # The core loop for agentic behavior. It iterates up to `max_iterations` times,
    # allowing the LLM to perform multiple steps of reasoning and tool use.
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1}/{max_iterations} ---")
        print("Sending request to LLM...")
        try:
            # Call the LLM with the current conversation history and available tools.
            # `tool_choice="auto"` allows the LLM to decide whether to call a tool or respond directly.
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools, # Pass the defined tools to the LLM
                tool_choice="auto",
                stream=False,
            )
        except Exception as e:
            # Handle potential errors during the LLM API call, e.g., Ollama not running.
            print(f"Error during Ollama API call: {e}")
            print(f"Ensure Ollama is running and the '{model_name}' model is pulled (run 'ollama run {model_name}' in your terminal).")
            return "An error occurred during processing."

        # Extract the LLM's response message and append it to the conversation history.
        # This is crucial for maintaining context across turns.
        response_message = response.choices[0].message
        messages.append(response_message)

        # Check if the LLM decided to call a tool.
        if response_message.tool_calls:
            # In this simplified example, we assume only one tool call per turn.
            tool_call = response_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            # If the LLM called the 'search' tool:
            if tool_name == "search":
                # Parse the 'query' argument provided by the LLM for the search.
                query_for_search = json.loads(tool_args).get("query")
                print(f"LLM called 'search' with query: '{query_for_search}'")

                # Execute the actual information retrieval using our `retrieve_info` function.
                retrieved_chunks = retrieve_info(query_for_search)
                context_str = "\n".join(retrieved_chunks)
                print("\nRetrieved Context:")
                for j, chunk in enumerate(retrieved_chunks):
                    print(f"  {j+1}. {chunk}")

                # Add the tool's output (the retrieved context) back to the conversation.
                # This allows the LLM to incorporate this new information in its subsequent reasoning.
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": context_str,
                    }
                )
            # If the LLM called the 'complete_response' tool:
            elif tool_name == "complete_response":
                # Extract the final answer from the LLM's tool call.
                final_answer = json.loads(tool_args).get("answer")
                print(f"LLM called 'complete_response' with answer:")
                print(final_answer)
                print("------------------------------------")
                return final_answer # Return the final answer and exit the pipeline.
            else:
                # Fallback for any unexpected tool calls.
                print(f"Unknown tool called: {tool_name}")
                return "An unexpected tool was called."
        else:
            # If the LLM did not call a tool, it means it provided a direct textual response.
            # In our agentic loop, this typically signifies that the LLM believes it has
            # sufficient information and is providing its final answer directly.
            print("LLM provided a direct response:")
            print(response_message.content)
            print("------------------------------------")
            return response_message.content # Return the direct response and exit.

    # If the loop completes without the LLM calling `complete_response` or providing a direct answer,
    # it means the maximum allowed iterations were reached without a conclusive result.
    print("Max iterations reached without a final answer.")
    return "Could not find a complete answer within the allowed iterations."
```

### 3. Integrating with Existing Components

We'll need to import `json` for parsing tool arguments and ensure our `retrieve_info` function is available. The `generate_response_with_ollama` function is no longer directly called by the pipeline but its core logic for interacting with Ollama is now embedded within the agentic loop's LLM calls.

## Complete Script

Here's the full script, combining the existing RAG setup with the new Agentic RAG pipeline.

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
import json # Added for parsing tool arguments

# 1. Define a small, in-memory knowledge base
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
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
    retrieved_chunks = [documents[idx] for idx in top_results.indices]
    return retrieved_chunks

# Define tools for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Searches the knowledge base for relevant information based on a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_response",
            "description": "Provides the final answer to the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete answer to the user's question.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

def agentic_rag_pipeline(user_query, max_iterations=5):
    """
    Orchestrates the Agentic RAG pipeline, allowing the LLM to use tools.
    This function manages the conversation flow, calls the LLM, and executes
    tools based on the LLM's decisions.
    """
    print(f"\n--- Processing Agentic Query: '{user_query}' ---")
    # Initialize the OpenAI client to interact with the Ollama LLM
    client = OpenAI(
        base_url="http://localhost:11434/v1", # Default Ollama API endpoint
        api_key="ollama", # Required, but can be any string for Ollama
    )
    model_name = "qwen3:4b" # Specify the model to use (ensure it's pulled in Ollama)

    # Initialize the conversation with a system message and the user's query.
    # The system message instructs the LLM on its capabilities and how to use the tools.
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You can search a knowledge base for information and provide a final answer. Use the 'search' tool if you need more information. Use the 'complete_response' tool when you have a complete answer to the user's question."},
        {"role": "user", "content": user_query}
    ]

    # The core loop for agentic behavior. It iterates up to `max_iterations` times,
    # allowing the LLM to perform multiple steps of reasoning and tool use.
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1}/{max_iterations} ---")
        print("Sending request to LLM...")
        try:
            # Call the LLM with the current conversation history and available tools.
            # `tool_choice="auto"` allows the LLM to decide whether to call a tool or respond directly.
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools, # Pass the defined tools to the LLM
                tool_choice="auto",
                stream=False,
            )
        except Exception as e:
            # Handle potential errors during the LLM API call, e.g., Ollama not running.
            print(f"Error during Ollama API call: {e}")
            print(f"Ensure Ollama is running and the '{model_name}' model is pulled (run 'ollama run {model_name}' in your terminal).")
            return "An error occurred during processing."

        # Extract the LLM's response message and append it to the conversation history.
        response_message = response.choices[0].message
        messages.append(response_message)

        # Check if the LLM decided to call a tool.
        if response_message.tool_calls:
            # Assuming one tool call per turn for simplicity in this example.
            tool_call = response_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            # Execute the 'search' tool if the LLM requested it.
            if tool_name == "search":
                # Parse the query argument from the LLM's tool call.
                query_for_search = json.loads(tool_args).get("query")
                print(f"LLM called 'search' with query: '{query_for_search}'")

                # Perform the actual information retrieval using our `retrieve_info` function.
                retrieved_chunks = retrieve_info(query_for_search)
                context_str = "\n".join(retrieved_chunks)
                print("\nRetrieved Context:")
                for j, chunk in enumerate(retrieved_chunks):
                    print(f"  {j+1}. {chunk}")

                # Add the tool's output (retrieved context) back to the conversation.
                # This allows the LLM to use the new information in subsequent turns.
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": context_str,
                    }
                )
            # Execute the 'complete_response' tool if the LLM decided to provide a final answer.
            elif tool_name == "complete_response":
                # Extract the final answer from the LLM's tool call.
                final_answer = json.loads(tool_args).get("answer")
                print(f"LLM called 'complete_response' with answer:")
                print(final_answer)
                print("------------------------------------")
                return final_answer # Return the final answer and exit the pipeline.
            else:
                # Handle cases where an unknown tool might be called (shouldn't happen with `tool_choice="auto"` and defined tools).
                print(f"Unknown tool called: {tool_name}")
                return "An unexpected tool was called."
        else:
            # If the LLM did not call a tool, it means it provided a direct textual response.
            # In our agentic loop, this typically means it's providing a final answer without explicitly
            # calling `complete_response`. We treat this as the final output.
            print("LLM provided a direct response:")
            print(response_message.content)
            print("------------------------------------")
            return response_message.content # Return the direct response and exit.

    # If the loop completes without the LLM calling `complete_response` or providing a direct answer,
    # it means the maximum iterations were reached without a conclusive result.
    print("Max iterations reached without a final answer.")
    return "Could not find a complete answer within the allowed iterations."

if __name__ == "__main__":
    # Example queries
    print("\n--- Running Agentic RAG Examples ---")
    agentic_rag_pipeline("When is the Innovation Summit held?")
    agentic_rag_pipeline("What is the Clarity Protocol?")
    agentic_rag_pipeline("Where can I find the best coffee machine?")
    agentic_rag_pipeline("What is the company's favorite animal?") # Query outside the context
    agentic_rag_pipeline("Tell me about the 'Quantum Leap' initiative and the 'Nexus Portal'.") # Multi-step query
```
