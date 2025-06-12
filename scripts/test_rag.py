import requests
import json
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()
BASE_URL = "http://127.0.0.1:8000"

def process_subtitles() -> Dict[str, Any]:
    """Process all subtitle files and create embeddings."""
    try:
        response = requests.post(f"{BASE_URL}/api/v1/subtitles/process")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error processing subtitles: {str(e)}[/red]")
        return None

def query_subtitles(query: str, n_results: int = 5, filter_episode: str = None) -> Dict[str, Any]:
    """Query the subtitle database."""
    try:
        params = {
            "query": query,
            "n_results": n_results
        }
        if filter_episode:
            params["filter_episode"] = filter_episode
        
        response = requests.post(f"{BASE_URL}/api/v1/subtitles/query", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error querying subtitles: {str(e)}[/red]")
        return None

def get_episodes() -> Dict[str, Any]:
    """Get list of all processed episodes."""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/subtitles/episodes")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Error getting episodes: {str(e)}[/red]")
        return None

def display_results(results: Dict[str, Any]):
    """Display query results in a formatted table."""
    if not results or "data" not in results:
        console.print("[red]No results found[/red]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Episode", style="dim")
    table.add_column("Timestamp")
    table.add_column("Dialogue", style="green")
    
    for item in results["data"]:
        table.add_row(
            f"Episode {item['episode_number']}",
            item['timestamp'],
            item['text']
        )
    
    console.print(table)

def main():
    # First, process the subtitles
    console.print("[bold blue]Processing subtitles...[/bold blue]")
    process_result = process_subtitles()
    if process_result:
        console.print(f"[green]Processed {process_result['data']['entries_processed']} subtitle entries[/green]")
    
    # Get list of episodes
    console.print("\n[bold blue]Getting episodes list...[/bold blue]")
    episodes = get_episodes()
    if episodes:
        console.print(f"[green]Found {len(episodes['data'])} episodes[/green]")
    
    # Example queries for English subtitles
    queries = [
        "What did Luffy say about becoming the Pirate King?",
        "Tell me about Zoro's dream to become the greatest swordsman",
        "What did Nami say about money and treasure?",
        "Find moments where Sanji talks about cooking",
        "Show me Usopp's lies about his adventures",
        "What did Chopper say about becoming a great doctor?",
        "Find Robin's quotes about history and knowledge",
        "Show me Franky's catchphrase 'SUPER!'",
        "What did Brook say about being a musician?",
        "Find moments where Jinbe talks about being a fishman"
    ]
    
    console.print("\n[bold blue]Testing queries...[/bold blue]")
    for i, query in enumerate(queries, 1):
        console.print(f"\n[bold yellow]Query {i}: {query}[/bold yellow]")
        results = query_subtitles(query)
        if results:
            display_results(results)

if __name__ == "__main__":
    main() 