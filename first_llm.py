import ollama

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {"role":"user", "content":"Apa perbedaan RAG dan fine-tuning?"}
    ]
)

print(response["message"]["content"])