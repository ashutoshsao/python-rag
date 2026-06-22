import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


api_key = get_required_env("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def read_file(path: str) -> str:
    file_path = Path(path)
    text = file_path.read_text()
    return text


def chunk_text(text: str, chunk_size: int, source: str) -> list[dict]:
    chunks: list[dict] = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]

        chunk_data = {
            "id": len(chunks),
            "text": chunk,
            "source": source,
            "start_from": i,
            "end_index": i + chunk_size,
        }

        chunks.append(chunk_data)

    return chunks


def score_chunk(query: str, chunk_text: str) -> int:
    query_words = query.lower().split()
    chunk_words = chunk_text.lower().split()

    score = 0

    for word in query_words:
        if word in chunk_words:
            score += 1

    return score


def search_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    scored_chunks: list[dict] = []

    for chunk in chunks:
        score = score_chunk(query, chunk["text"])

        if score > 0:
            result = {
                "id": chunk["id"],
                "text": chunk["text"],
                "score": score,
                "source": chunk["source"],
            }

            scored_chunks.append(result)

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    return scored_chunks[:top_k]


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts: list[str] = []

    for chunk in chunks:
        context_part = f"""
Source: {chunk["source"]}
Chunk ID: {chunk["id"]}
Content:
{chunk["text"]}
"""
        context_parts.append(context_part)

    context = "\n---\n".join(context_parts)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the context below.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

User question:
{query}

Answer:
"""

    return prompt


def main() -> None:
    source = "data/python.txt"
    text = read_file(source)

    chunks = chunk_text(text, 100, source)

    query = "why use python for learning applied ai engineering"

    results = search_chunks(query, chunks, 3)

    print("Query", query)
    print("-----------")

    prompt = build_prompt(
        query,
        results,
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    print(response.text)


main()
