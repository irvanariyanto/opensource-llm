try:
    from .rag_chatbot import chat
except ImportError:
    from rag_chatbot import chat


def main() -> None:
    print("\n=== RAG Chatbot Ready! Type 'exit' to quit ===\n")
    while True:
        question = input("Pertanyaan: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "/bye"}:
            break

        result = chat(question)
        print(f"\nJawaban:\n{result['answer']}\n")
        print("Sumber:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] Halaman {src['page']}: {src['snippet']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
