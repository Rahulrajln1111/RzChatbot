import os

from rag import MODEL_CACHE_DIR, RERANK_MODEL


def main() -> None:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    print(f"Downloading {RERANK_MODEL} -> {MODEL_CACHE_DIR}")

    from sentence_transformers import CrossEncoder

    CrossEncoder(RERANK_MODEL)
    print("Done.")


if __name__ == "__main__":
    main()
