import requests
import json
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"

def process_subtitles() -> Dict[str, Any]:
    """Process all subtitle files and create embeddings."""
    response = requests.post(f"{BASE_URL}/api/v1/subtitles/process")
    return response.json()

def query_subtitles(query: str, n_results: int = 5, filter_episode: str = None) -> Dict[str, Any]:
    """Query the subtitle database."""
    params = {
        "query": query,
        "n_results": n_results
    }
    if filter_episode:
        params["filter_episode"] = filter_episode
    
    response = requests.post(f"{BASE_URL}/api/v1/subtitles/query", params=params)
    return response.json()

def get_episodes() -> Dict[str, Any]:
    """Get list of all processed episodes."""
    response = requests.get(f"{BASE_URL}/api/v1/subtitles/episodes")
    return response.json()

def main():
    # First, process the subtitles
    print("Processing subtitles...")
    process_result = process_subtitles()
    print(json.dumps(process_result, indent=2))
    
    # Get list of episodes
    print("\nGetting episodes list...")
    episodes = get_episodes()
    print(json.dumps(episodes, indent=2))
    
    # Example queries
    queries = [
        "What did Luffy say about becoming the Pirate King?",
        "Tell me about Zoro's dream",
        "What did Nami say about money?"
    ]
    
    print("\nTesting queries...")
    for query in queries:
        print(f"\nQuery: {query}")
        results = query_subtitles(query)
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 