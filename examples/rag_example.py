#!/usr/bin/env python3
"""RAG (Retrieval-Augmented Generation) Example using NPU Proxy

This example demonstrates how to use the NPU Proxy's embeddings and chat
endpoints to build a simple RAG system.

Requirements:
    pip install openai numpy

Usage:
    # Start the NPU proxy first:
    # Windows: python -m uvicorn npu_proxy.main:app --host 0.0.0.0 --port 11435
    
    # Then run this example:
    python examples/rag_example.py
"""

import numpy as np
from openai import OpenAI

# Configure client to use NPU Proxy
client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="not-needed",  # NPU Proxy doesn't require auth
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))


def embed_text(text: str) -> list[float]:
    """Get embedding for a single text."""
    response = client.embeddings.create(
        model="all-minilm-l6-v2",
        input=text,
    )
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    response = client.embeddings.create(
        model="all-minilm-l6-v2",
        input=texts,
    )
    return [item.embedding for item in response.data]


def find_most_similar(query_embedding: list[float], 
                      doc_embeddings: list[list[float]], 
                      documents: list[str],
                      top_k: int = 2) -> list[tuple[str, float]]:
    """Find the top-k most similar documents to the query."""
    similarities = [
        (doc, cosine_similarity(query_embedding, emb))
        for doc, emb in zip(documents, doc_embeddings)
    ]
    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def rag_query(query: str, documents: list[str], doc_embeddings: list[list[float]]) -> str:
    """Perform RAG: retrieve relevant docs and generate answer."""
    # Step 1: Embed the query
    query_embedding = embed_text(query)
    
    # Step 2: Find most similar documents
    similar_docs = find_most_similar(query_embedding, doc_embeddings, documents, top_k=2)
    
    print(f"\n📚 Retrieved documents:")
    for doc, score in similar_docs:
        print(f"  - [{score:.3f}] {doc[:60]}...")
    
    # Step 3: Build context from retrieved documents
    context = "\n".join([doc for doc, _ in similar_docs])
    
    # Step 4: Generate answer using chat completion
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions based on the provided context. Be concise."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        }
    ]
    
    response = client.chat.completions.create(
        model="tinyllama-1.1b-chat-int4-ov",
        messages=messages,
        max_tokens=100,
        temperature=0.7,
    )
    
    return response.choices[0].message.content


def main():
    """Run the RAG example."""
    print("🚀 NPU Proxy RAG Example")
    print("=" * 50)
    
    # Sample knowledge base
    documents = [
        "The Intel Neural Processing Unit (NPU) is designed for AI workloads with high efficiency and low power consumption.",
        "WSL2 runs a full Linux kernel in a lightweight virtual machine, enabling native Linux compatibility on Windows.",
        "OpenVINO is Intel's toolkit for optimizing and deploying AI models on Intel hardware including CPUs, GPUs, and NPUs.",
        "The NPU Proxy enables WSL2 applications to access the Windows host's NPU via an OpenAI-compatible API.",
        "TinyLlama is a compact 1.1B parameter language model that can run efficiently on edge devices.",
        "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI documentation.",
        "Embeddings are dense vector representations of text that capture semantic meaning for similarity search.",
        "Python's asyncio enables concurrent programming with async/await syntax for I/O-bound operations.",
    ]
    
    print(f"\n📖 Knowledge base: {len(documents)} documents")
    
    # Step 1: Embed all documents
    print("\n⏳ Embedding documents...")
    doc_embeddings = embed_batch(documents)
    print(f"✅ Generated {len(doc_embeddings)} embeddings (dim={len(doc_embeddings[0])})")
    
    # Step 2: Run some queries
    queries = [
        "What is the NPU used for?",
        "How does WSL2 work?",
        "What can I do with OpenVINO?",
    ]
    
    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"❓ Query: {query}")
        
        answer = rag_query(query, documents, doc_embeddings)
        print(f"\n💡 Answer: {answer}")
    
    print(f"\n{'=' * 50}")
    print("✅ RAG example complete!")


if __name__ == "__main__":
    main()
