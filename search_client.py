import httpx
from dotenv import load_dotenv
import os

load_dotenv()
def search_internet(query: str, max_results: int = 5):
    params = {
        "q": query,
        "format": "json",
    }

    with httpx.Client(timeout=15.0) as client:
        response = client.get(
            f"{os.environ['SEARXNG_URL']}/search",
            params=params
        )
        response.raise_for_status()

        data = response.json()

    return [
        {
            "title": item["title"],
            "url": item["url"],
            "content": item["content"],
        }
        for item in data.get("results", [])[:max_results]
    ]