import os
import sys
import time
import urllib.request
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from llama_cpp import Llama

# --- CONFIGURATION ---
MODEL_URL = os.environ.get("ASC_MODEL_URL", "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf")
MODEL_NAME = os.environ.get("ASC_MODEL_NAME", "Phi-3-mini-4k-instruct-q4.gguf")
CONSOLE = Console()

# --- CLEAN ASCII THEME ---
ART = """
   _____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____ 
  /  _  ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ |
 |  | | ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ |
 |  | | ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ |
 |  |_| ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ |
 |  _  ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ |
 |  | | ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ |
 |  |_| ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ ||  _ |
 |______||____||____||____||____||____||____||____||____||____||____||____|
"""

# --- SETUP ---
def setup():
    CONSOLE.print(Panel(ART, title="ASC TERMINAL", border_style="cyan"))
    CONSOLE.print("[bold green]Initializing System...[/bold green]")
    
    # Check Model
    if not os.path.exists(MODEL_NAME):
        CONSOLE.print("[yellow]Downloading AI Model (2.3GB)...[/yellow]")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
            CONSOLE.print("[green]Model Downloaded.[/green]")
        except Exception as e:
            CONSOLE.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    else:
        CONSOLE.print("[green]Model Found.[/green]")

    # Check Dependencies
    try:
        import rich
        import llama_cpp
    except ImportError:
        CONSOLE.print("[yellow]Installing Dependencies...[/yellow]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "llama-cpp-python", "--quiet"])
        CONSOLE.print("[green]Dependencies Installed.[/green]")

    # Load Model
    CONSOLE.print("[bold blue]Loading AI Engine...[/bold blue]")
    try:
        llm = Llama(model_path=MODEL_NAME, n_ctx=2048, n_threads=4)
        CONSOLE.print("[green]System Ready.[/green]")
        return llm
    except Exception as e:
        CONSOLE.print(f"[red]Load Failed: {e}[/red]")
        sys.exit(1)

# --- AI LOGIC ---
def get_ai_response(llm, user_input, history):
    system_prompt = """You are ASC Terminal. 
    1. Speak in straight English.
    2. Recommend actions based on user intent.
    3. No hallucinations. If unsure, say 'Unknown'.
    4. Keep responses short and actionable."""
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]
    output = llm.create_chat_completion(messages=messages, temperature=0.3)
    return output['choices'][0]['message']['content']

# --- MAIN LOOP ---
def main():
    llm = setup()
    history = []
    
    CONSOLE.print("\n[bold]Type 'exit' to quit.[/bold]\n")
    
    while True:
        try:
            user_input = Prompt.ask("[cyan]ASC[/cyan] [white]>>[/white]")
            if user_input.lower() in ['exit', 'quit']:
                CONSOLE.print("[yellow]Shutting down...[/yellow]")
                break
            
            CONSOLE.print("[dim]Processing...[/dim]")
            response = get_ai_response(llm, user_input, history)
            
            CONSOLE.print(Panel(response, title="ASC Response", border_style="green"))
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            CONSOLE.print("\n[yellow]Interrupted.[/yellow]")
            break

if __name__ == "__main__":
    main()
