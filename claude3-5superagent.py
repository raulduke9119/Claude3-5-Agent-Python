import json
import base64
import io
import re
import time
import os
import subprocess
from typing import List, Optional, Dict, Any, Tuple

import nltk
import requests
import pylint.lint
import autopep8
from PIL import Image
from bs4 import BeautifulSoup
from anthropic import Anthropic, APIStatusError, APIError
from tavily import TavilyClient
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from pydantic import BaseModel, Field
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Constants
CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
MAX_CONTINUATION_ITERATIONS = 25
MAINMODEL = "claude-3-5-sonnet-20240620"
TOOLCHECKERMODEL = "claude-3-5-sonnet-20240620"
MAX_WEBPAGE_SUMMARY_TOKENS = 1024

# Initialize clients and resources
console = Console()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Download NLTK data
nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words', 'vader_lexicon'])

# Initialize NLP tools
sia = SentimentIntensityAnalyzer()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Conversation state
conversation_history: List[Dict[str, Any]] = []
automode = False

# Pydantic models
class Author(BaseModel):
    name: str
    email: str

class Note(BaseModel):
    content: str
    author: Author
    tags: Optional[List[str]] = None
    priority: int = Field(ge=1, le=5, default=3)

class SentimentScores(BaseModel):
    positive: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)

# System prompts
base_system_prompt = """
You are Claude, an AI assistant powered by Anthropic's Claude-3.5-Sonnet model, specializing in software development. Your capabilities include:

1. Creating and managing project structures
2. Writing, debugging, and improving code across multiple languages
3. Providing architectural insights and applying design patterns
4. Staying current with the latest technologies and best practices
5. Analyzing and manipulating files within the project directory
6. Performing web searches for up-to-date information
7. Performing calculations and structured data analysis
8. Executing shell commands for various operations

Available tools and their optimal use cases:

1. create_folder: Create new directories in the project structure.
2. create_file: Generate new files with specified content.
3. edit_and_apply: Examine and modify existing files.
4. read_file: View the contents of existing files without making changes.
5. list_files: Understand the current project structure or locate specific files.
6. tavily_search: Obtain current information on technologies, libraries, or best practices.
7. calculator: Perform basic arithmetic operations.
8. sentiment_analysis: Analyze the sentiment of given text.
9. entity_extraction: Extract named entities from text.
10. summarize: Generate a concise summary of given text.
11. summarize_webpage: Fetch and summarize the content of a webpage.
12. pylint_check: Run Pylint on Python code to check for errors and style issues.
13. autopep8_format: Format Python code according to PEP 8 style guide.
14. shell_command: Execute shell commands like curl and other similar operations.

Tool Usage Guidelines:
- Always use the most appropriate tool for the task at hand.
- For file modifications, use edit_and_apply. Read the file first, then apply changes if needed.
- After making changes, always review the diff output to ensure accuracy.
- Proactively use tavily_search when you need up-to-date information or context.
- Use structured data tools (sentiment_analysis, entity_extraction, summarize) when appropriate.
- Use pylint_check and autopep8_format for Python code quality and formatting.
- Use shell_command for system-level operations when necessary, but be cautious with its usage.

Error Handling and Recovery:
- If a tool operation fails, analyze the error message and attempt to resolve the issue.
- For file-related errors, check file paths and permissions before retrying.
- If a search fails, try rephrasing the query or breaking it into smaller, more specific searches.
- For shell commands, ensure proper syntax and handle potential security implications.

Always strive for accuracy, clarity, and efficiency in your responses and actions. If uncertain, use the tavily_search tool or admit your limitations.
"""

automode_system_prompt = """
You are currently in automode. Follow these guidelines:

1. Goal Setting:
   - Set clear, achievable goals based on the user's request.
   - Break down complex tasks into smaller, manageable goals.

2. Goal Execution:
   - Work through goals systematically, using appropriate tools for each task.
   - Utilize file operations, code writing, web searches, and shell commands as needed.
   - Always read a file before editing and review changes after editing.

3. Progress Tracking:
   - Provide regular updates on goal completion and overall progress.
   - Use the iteration information to pace your work effectively.

4. Tool Usage:
   - Leverage all available tools to accomplish your goals efficiently.
   - Prefer edit_and_apply for file modifications, applying changes in chunks for large edits.
   - Use tavily_search proactively for up-to-date information.
   - Utilize shell_command for system-level operations when necessary.

5. Error Handling:
   - If a tool operation fails, analyze the error and attempt to resolve the issue.
   - For persistent errors, consider alternative approaches to achieve the goal.

6. Automode Completion:
   - When all goals are completed, respond with "AUTOMODE_COMPLETE" to exit automode.
   - Do not ask for additional tasks or modifications once goals are achieved.

7. Iteration Awareness:
   - You have access to this {iteration_info}.
   - Use this information to prioritize tasks and manage time effectively.

Remember: Focus on completing the established goals efficiently and effectively. Avoid unnecessary conversations or requests for additional tasks.
"""

def update_system_prompt(current_iteration: Optional[int] = None, max_iterations: Optional[int] = None) -> str:
    chain_of_thought_prompt = """
    Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within <thinking></thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.

    Do not reflect on the quality of the returned search results in your response.
    """
    if automode:
        iteration_info = f"You are currently on iteration {current_iteration} out of {max_iterations} in automode." if current_iteration is not None and max_iterations is not None else ""
        return f"{base_system_prompt}\n\n{automode_system_prompt.format(iteration_info=iteration_info)}\n\n{chain_of_thought_prompt}"
    else:
        return f"{base_system_prompt}\n\n{chain_of_thought_prompt}"

def create_folder(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as e:
        return f"Error creating folder: {str(e)}"

def create_file(path: str, content: str = "") -> str:
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"File created: {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

def highlight_diff(diff_text: str) -> Syntax:
    return Syntax(diff_text, "diff", theme="monokai", line_numbers=True)

def generate_and_apply_diff(original_content: str, new_content: str, path: str) -> str:
    import difflib
    diff = list(difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3
    ))

    if not diff:
        return "No changes detected."

    try:
        with open(path, 'w') as f:
            f.writelines(new_content)

        diff_text = ''.join(diff)
        highlighted_diff = highlight_diff(diff_text)

        diff_panel = Panel(
            highlighted_diff,
            title=f"Changes in {path}",
            expand=False,
            border_style="cyan"
        )

        console.print(diff_panel)

        added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

        summary = f"Changes applied to {path}:\n"
        summary += f"  Lines added: {added_lines}\n"
        summary += f"  Lines removed: {removed_lines}\n"

        return summary

    except Exception as e:
        error_panel = Panel(
            f"Error: {str(e)}",
            title="Error Applying Changes",
            style="bold red"
        )
        console.print(error_panel)
        return f"Error applying changes: {str(e)}"

def edit_and_apply(path: str, new_content: str) -> str:
    try:
        with open(path, 'r') as file:
            original_content = file.read()
        
        if new_content != original_content:
            diff_result = generate_and_apply_diff(original_content, new_content, path)
            return f"Changes applied to {path}:\n{diff_result}"
        else:
            return f"No changes needed for {path}"
    except Exception as e:
        return f"Error editing/applying to file: {str(e)}"

def read_file(path: str) -> str:
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def list_files(path: str = ".") -> str:
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def tavily_search(query: str) -> Dict[str, Any]:
    try:
        response = tavily.search(query=query, search_depth="advanced")
        if isinstance(response, dict):
            return response
        else:
            return {"error": "Unexpected response format", "details": str(response)}
    except Exception as e:
        return {"error": "Error performing search", "details": str(e)}


def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def sentiment_analysis(text: str) -> Dict[str, float]:
    sentiment_scores = sia.polarity_scores(text)
    return {
        "positive": sentiment_scores['pos'],
        "negative": sentiment_scores['neg'],
        "neutral": sentiment_scores['neu'],
        "compound": sentiment_scores['compound']
    }

def entity_extraction(text: str) -> Dict[str, List[Dict[str, str]]]:
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)
    entities = []
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity_type = subtree.label()
            entity_value = " ".join([word for word, tag in subtree.leaves()])
            entities.append({"type": entity_type, "value": entity_value})
    return {"entities": entities}

def summarize(text: str) -> Dict[str, str]:
    max_length = min(len(text.split()) // 2, 150)  # Limit summary to half the original length or 150 words
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
    return {"summary": summary}

def summarize_webpage(url: str) -> str:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            page_content = soup.get_text()
            
            prompt = f"<content>{page_content}</content>Please produce a concise summary of the web page content."
            
            summary, _ = chat_with_claude(prompt)
            
            return summary
        else:
            return f"Failed to fetch the web page. Status code: {response.status_code}"
    except Exception as e:
        return f"Error summarizing webpage: {str(e)}"

def pylint_check(code: str) -> str:
    try:
        with open('temp_code.py', 'w') as f:
            f.write(code)
        
        pylint_output = io.StringIO()
        pylint.lint.Run(['temp_code.py'], do_exit=False, reporter=pylint.reporters.text.TextReporter(pylint_output))
        
        os.remove('temp_code.py')
        
        return pylint_output.getvalue()
    except Exception as e:
        return f"Error running Pylint: {str(e)}"

def autopep8_format(code: str) -> str:
    try:
        formatted_code = autopep8.fix_code(code)
        return formatted_code
    except Exception as e:
        return f"Error formatting code: {str(e)}"

def shell_command(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"

def encode_image_to_base64(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.DEFAULT_STRATEGY)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        return f"Error encoding image: {str(e)}"

def parse_goals(response: str) -> List[str]:
    return re.findall(r'Goal \d+: (.+)', response)

def execute_goals(goals: List[str]) -> None:
    global automode
    for i, goal in enumerate(goals, 1):
        console.print(Panel(f"Executing Goal {i}: {goal}", title="Goal Execution", style="bold yellow"))
        response, _ = chat_with_claude(f"Continue working on goal: {goal}")
        if CONTINUATION_EXIT_PHRASE in response:
            automode = False
            console.print(Panel("Exiting automode.", title="Automode", style="bold green"))
            break

tools = [
    {
        "name": "create_folder",
        "description": "Create a new folder at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path where the folder should be created"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "create_file",
        "description": "Create a new file at the specified path with content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path where the file should be created"},
                "content": {"type": "string", "description": "The content of the file"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_and_apply",
        "description": "Apply changes to a file. Provide the full file content when editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path of the file to edit"},
                "new_content": {"type": "string", "description": "The new content to apply to the file"}
            },
            "required": ["path", "new_content"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path of the file to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files and directories in the specified folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path of the folder to list (default: current directory)"}
            }
        }
    },
  {
        "name": "tavily_search",
        "description": "Perform a web search using Tavily API to get up-to-date information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    },
    
    {
        "name": "calculator",
        "description": "Perform basic arithmetic operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The mathematical expression to evaluate"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "sentiment_analysis",
        "description": "Analyze the sentiment of given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "entity_extraction",
        "description": "Extract named entities from text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to extract entities from"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "summarize",
        "description": "Generate a concise summary of given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to summarize"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "summarize_webpage",
        "description": "Fetch and summarize the content of a webpage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL of the webpage to summarize"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "pylint_check",
        "description": "Run Pylint on Python code to check for errors and style issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python code to check"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "autopep8_format",
        "description": "Format Python code according to PEP 8 style guide.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python code to format"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "shell_command",
        "description": "Execute shell commands like curl and other similar operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"}
            },
            "required": ["command"]
        }
    }
]



def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Any:
    try:
        if tool_name == "create_folder":
            return create_folder(tool_input["path"])
        elif tool_name == "create_file":
            return create_file(tool_input["path"], tool_input.get("content", ""))
        elif tool_name == "edit_and_apply":
            return edit_and_apply(tool_input["path"], tool_input["new_content"])
        elif tool_name == "read_file":
            return read_file(tool_input["path"])
        elif tool_name == "list_files":
            return list_files(tool_input.get("path", "."))
        elif tool_name == "tavily_search":
            result = tavily_search(tool_input["query"])
            return json.dumps(result)  # Ensure the result is a JSON string
        elif tool_name == "calculator":
            return calculator(tool_input["expression"])
        elif tool_name == "sentiment_analysis":
            return sentiment_analysis(tool_input["text"])
        elif tool_name == "entity_extraction":
            return entity_extraction(tool_input["text"])
        elif tool_name == "summarize":
            return summarize(tool_input["text"])
        elif tool_name == "summarize_webpage":
            return summarize_webpage(tool_input["url"])
        elif tool_name == "pylint_check":
            return pylint_check(tool_input["code"])
        elif tool_name == "autopep8_format":
            return autopep8_format(tool_input["code"])
        elif tool_name == "shell_command":
            return shell_command(tool_input["command"])
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except KeyError as e:
        return json.dumps({"error": f"Missing required parameter {str(e)} for tool {tool_name}"})
    except Exception as e:
        return json.dumps({"error": f"Error executing tool {tool_name}: {str(e)}"})


def chat_with_claude(user_input: str, image_path: Optional[str] = None, current_iteration: Optional[int] = None, max_iterations: Optional[int] = None) -> Tuple[str, bool]:
    global conversation_history, automode

    current_conversation = []

    if image_path:
        console.print(Panel(f"Processing image at path: {image_path}", title_align="left", title="Image Processing", expand=False, style="yellow"))
        image_base64 = encode_image_to_base64(image_path)

        if image_base64.startswith("Error"):
            console.print(Panel(f"Error encoding image: {image_base64}", title="Error", style="bold red"))
            return "I'm sorry, there was an error processing the image. Please try again.", False

        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": f"User input for image: {user_input}"
                }
            ]
        }
        current_conversation.append(image_message)
        console.print(Panel("Image message added to conversation history", title_align="left", title="Image Added", style="green"))
    else:
        current_conversation.append({"role": "user", "content": user_input})

    messages = conversation_history + current_conversation

    try:
        response = client.messages.create(
            model=MAINMODEL,
            max_tokens=4000,
            system=update_system_prompt(current_iteration, max_iterations),
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"}
        )
    except APIStatusError as e:
        if e.status_code == 429:
            console.print(Panel("Rate limit exceeded. Retrying after a short delay...", title="API Error", style="bold yellow"))
            time.sleep(5)
            return chat_with_claude(user_input, image_path, current_iteration, max_iterations)
        else:
            console.print(Panel(f"API Error: {str(e)}", title="API Error", style="bold red"))
            return "I'm sorry, there was an error communicating with the AI. Please try again.", False
    except APIError as e:
        console.print(Panel(f"API Error: {str(e)}", title="API Error", style="bold red"))
        return "I'm sorry, there was an error communicating with the AI. Please try again.", False

    assistant_response = ""
    exit_continuation = False
    tool_uses = []

    for content_block in response.content:
        if content_block.type == "text":
            assistant_response += content_block.text
            if CONTINUATION_EXIT_PHRASE in content_block.text:
                exit_continuation = True
        elif content_block.type == "tool_use":
            tool_uses.append(content_block)

    console.print(Panel(Markdown(assistant_response), title="Claude's Response", title_align="left", expand=False))

    for tool_use in tool_uses:
        tool_name = tool_use.name
        tool_input = tool_use.input
        tool_use_id = tool_use.id

        console.print(Panel(f"Tool Used: {tool_name}", style="green"))
        console.print(Panel(f"Tool Input: {json.dumps(tool_input, indent=2)}", style="green"))

        try:
            result = execute_tool(tool_name, tool_input)
            console.print(Panel(str(result), title_align="left", title="Tool Result", style="green"))
        except Exception as e:
            result = f"Error executing tool: {str(e)}"
            console.print(Panel(result, title="Tool Execution Error", style="bold red"))
        
        current_conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": tool_input
                }
            ]
        })

        current_conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result
                }
            ]
        })

        messages = conversation_history + current_conversation

        try:
            tool_response = client.messages.create(
                model=TOOLCHECKERMODEL,
                max_tokens=4000,
                system=update_system_prompt(current_iteration, max_iterations),
                messages=messages,
                tools=tools,
                tool_choice={"type": "auto"}
            )

            tool_checker_response = ""
            for tool_content_block in tool_response.content:
                if tool_content_block.type == "text":
                    tool_checker_response += tool_content_block.text
            console.print(Panel(Markdown(tool_checker_response), title="Claude's Response to Tool Result", title_align="left"))
            assistant_response += "\n\n" + tool_checker_response
        except APIError as e:
            error_message = f"Error in tool response: {str(e)}"
            console.print(Panel(error_message, title="Error", style="bold red"))
            assistant_response += f"\n\n{error_message}"

    if assistant_response:
        current_conversation.append({"role": "assistant", "content": assistant_response})

    conversation_history = messages + [{"role": "assistant", "content": assistant_response}]

    return assistant_response, exit_continuation

def handle_image_input() -> Tuple[Optional[str], Optional[bool]]:
    image_path = console.input("[bold cyan]Drag and drop your image here, then press enter:[/bold cyan] ").strip().replace("'", "")
    if os.path.isfile(image_path):
        user_input = console.input("[bold cyan]You (prompt for image):[/bold cyan] ")
        return chat_with_claude(user_input, image_path)
    else:
        console.print(Panel("Invalid image path. Please try again.", title="Error", style="bold red"))
        return None, None

def handle_automode(user_input: str) -> None:
    global automode, conversation_history
    parts = user_input.split()
    max_iterations = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else MAX_CONTINUATION_ITERATIONS

    automode = True
    console.print(Panel(f"Entering automode with {max_iterations} iterations. Please provide the goal of the automode.", title_align="left", title="Automode", style="bold yellow"))
    console.print(Panel("Press Ctrl+C at any time to exit the automode loop.", style="bold yellow"))
    user_input = console.input("[bold cyan]You:[/bold cyan] ")

    iteration_count = 0
    try:
        while automode and iteration_count < max_iterations:
            response, exit_continuation = chat_with_claude(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations)

            if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                console.print(Panel("Automode completed.", title_align="left", title="Automode", style="green"))
                automode = False
            else:
                console.print(Panel(f"Continuation iteration {iteration_count + 1} completed. Press Ctrl+C to exit automode. ", title_align="left", title="Automode", style="yellow"))
            
            user_input = "Continue with the next step. Or STOP by saying 'AUTOMODE_COMPLETE' if you think you've achieved the results."
            iteration_count += 1

    except KeyboardInterrupt:
        handle_automode_interrupt()

def handle_automode_interrupt() -> None:
    global automode
    automode = False
    console.print(Panel("Automode interrupted. Returning to regular chat.", style="yellow"))

def main() -> None:
    global automode, conversation_history
    console.print("Type 'automode [number]' to enter Autonomous mode with a specific number of iterations.")
    console.print("Type 'summarize http://example.com' to summarize a webpage.")
    console.print("Type 'shell [command]' to execute a shell command.")
    console.print("While in automode, press Ctrl+C at any time to exit the automode to return to regular chat.")

    while True:
        user_input = console.input("[bold cyan]You:[/bold cyan] ")

        if user_input.lower() == 'exit':
            console.print(Panel("Thank you for chatting. Goodbye!", title_align="left", title="Goodbye", style="bold green"))
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
            console.print(Panel("Exited automode. Returning to regular chat.", style="green"))
        elif user_input.lower().startswith('summarize http'):
            url = user_input.split(' ', 1)[1]
            response = summarize_webpage(url)
            console.print(Panel(response, title="Webpage Summary", style="cyan"))
        elif user_input.lower().startswith('shell'):
            command = user_input.split(' ', 1)[1]
            response = shell_command(command)
            console.print(Panel(response, title="Shell Command Result", style="cyan"))
        else:
            response, _ = chat_with_claude(user_input)

if __name__ == "__main__":
    main()
