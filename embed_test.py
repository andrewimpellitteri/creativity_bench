
import ollama

try:
    embedding = ollama.embeddings(
        model='snowflake-arctic-embed',
        prompt='test'
    )
    print("Success:", len(embedding.embedding))
except Exception as e:
    print("Error:", e)