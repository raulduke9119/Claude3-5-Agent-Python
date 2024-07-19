import os
from dotenv import load_dotenv
from rich.console import Console
from anthropic import Anthropic
from toolsutils import (
    chat_with_claude, handle_image_input, handle_automode,
    handle_automode_interrupt, summarize_webpage, shell_command
)
from rich.table import Table
from rich.panel import Panel
# Load environment variables
load_dotenv(override=True)

# Initialize clients and resources
console = Console()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))



def main():
    table = Table(title="Available Tools", show_header=True, header_style="bold magenta")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")

    tools = [
        ("create_folder", "Create new directories"),
        ("create_file", "Generate new files with content"),
        ("edit_and_apply", "Modify existing files"),
        ("read_file", "View file contents"),
        ("list_files", "List files in a directory"),
        ("tavily_search", "Perform web searches"),
        ("calculator", "Perform arithmetic operations"),
        ("sentiment_analysis", "Analyze text sentiment"),
        ("entity_extraction", "Extract named entities"),
        ("summarize", "Generate text summaries"),
        ("summarize_webpage", "Summarize webpage content"),
        ("pylint_check", "Check Python code quality"),
        ("autopep8_format", "Format Python code"),
        ("shell_command", "Execute shell commands")
    ]

    for tool, description in tools:
        table.add_row(tool, description)

    console.print(table)

    instructions = Panel(
        "[bold yellow]Special Commands:[/bold yellow]\n"
        "• Type [cyan]'automode [number]'[/cyan] to enter Autonomous mode with a specific number of iterations.\n"
        "• While in automode, press [red]Ctrl+C[/red] at any time to exit and return to regular chat.",
        title="Instructions",
        expand=False,
        border_style="bold blue"
    )
    console.print(instructions)



    while True:
        user_input = console.input("[bold cyan]You:[/bold cyan] ")

        if user_input.lower() == 'exit':
            console.print("Thank you for chatting. Goodbye!")
            break

        if user_input.lower() == 'image':
            response, _ = handle_image_input()
            if response:
                continue
        elif user_input.lower().startswith('automode'):
            try:
                handle_automode(user_input)
            except KeyboardInterrupt:
                handle_automode_interrupt()
            console.print("Exited automode. Returning to regular chat.")
        elif user_input.lower().startswith('summarize http'):
            url = user_input.split(' ', 1)[1]
            response = summarize_webpage(url)
            console.print(response)
        elif user_input.lower().startswith('shell'):
            command = user_input.split(' ', 1)[1]
            response = shell_command(command)
            console.print(response)
        else:
            response, _ = chat_with_claude(user_input)

if __name__ == "__main__":
    main()
