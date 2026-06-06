import ollama

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {"role":"system", "content":"Kamu adalah guru fisika yang menjelaskan dengan analogi sederhana."},
        {"role":"user", "content":"Jelaskan teori relativitas"}
    ]
)

print(response["message"]["content"])