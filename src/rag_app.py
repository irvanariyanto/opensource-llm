"""CLI entry point for the RAG chatbot."""

try:
    from .rag_chatbot import chat
except ImportError:
    from rag_chatbot import chat


def main() -> None:
    """Run an interactive question-answer loop in the terminal."""
    print("\n=== RAG Chatbot Ready! Type 'exit' to quit ===\n")

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "/bye"}:
            break

        result = chat(question)
        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources:")
        for i, src in enumerate(result["sources"], 1):
            score = src.get("score", "")
            score_str = f" (score: {score})" if score else ""
            print(f"  [{i}] Page {src['page']}: {src['snippet']}{score_str}")
        print("-" * 60)


if __name__ == "__main__":
    main()
