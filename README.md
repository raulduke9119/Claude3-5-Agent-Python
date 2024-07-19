# AI Assistant with Claude

This project is an advanced command-line interface that harnesses the power of Anthropic's Claude AI model to provide a wide range of functionalities. It's designed to assist developers, researchers, and curious minds with tasks ranging from code assistance to web summarization and autonomous task execution.

## Features

### 1. Interactive AI Chat
Engage in dynamic conversations with Claude, leveraging its vast knowledge base for problem-solving, brainstorming, and information retrieval.

### 2. Autonomous Mode
Enter an extended task execution mode where Claude can work on complex, multi-step problems without constant user input.

### 3. Web Intelligence
- **Web Page Summarization**: Quickly distill the key points from any web page.
- **Web Searches**: Utilize the Tavily search API for up-to-date information retrieval.

### 4. Code Assistance
- **Code Quality Checks**: Run Pylint on Python code to identify and fix issues.
- **Code Formatting**: Automatically format Python code to PEP 8 standards using autopep8.
- **File and Folder Management**: Create, read, edit, and list files and folders in your project structure.

### 5. Natural Language Processing
- **Sentiment Analysis**: Analyze the emotional tone of text.
- **Entity Extraction**: Identify and extract named entities from text.
- **Text Summarization**: Generate concise summaries of longer text passages.

### 6. System Integration
- **Shell Command Execution**: Run shell commands directly from the interface.

### 7. Data Processing
- **Calculator Functionality**: Perform quick calculations without leaving the interface.

## Available Tools

1. `create_folder`: Create new directories in your project structure.
2. `create_file`: Generate new files with specified content.
3. `edit_and_apply`: Examine and modify existing files with diff output.
4. `read_file`: View the contents of existing files.
5. `list_files`: Understand the current project structure or locate specific files.
6. `tavily_search`: Obtain current information on technologies, libraries, or best practices.
7. `calculator`: Perform basic arithmetic operations.
8. `sentiment_analysis`: Analyze the sentiment of given text.
9. `entity_extraction`: Extract named entities from text.
10. `summarize`: Generate a concise summary of given text.
11. `summarize_webpage`: Fetch and summarize the content of a webpage.
12. `pylint_check`: Run Pylint on Python code to check for errors and style issues.
13. `autopep8_format`: Format Python code according to PEP 8 style guide.
14. `shell_command`: Execute shell commands for system-level operations.

## Getting Started

Please refer to the [Getting Started](GETTING_STARTED.md) guide for detailed installation and usage instructions.

## Requirements

This project requires Python 3.7 or higher. For a complete list of Python dependencies, please see the [requirements.txt](requirements.txt) file.

## Usage

After installation, run the main script:


You'll be presented with a table of available tools and instructions for special commands. Interact with the AI assistant by typing your queries or commands.

### Special Commands

- `automode [number]`: Enter Autonomous mode with a specified number of iterations.
- `Ctrl+C`: While in automode, use this to exit and return to regular chat.
- `exit`: End the session and close the application.

## Configuration

The project uses environment variables for configuration. Create a `.env` file in the project root with the following structure:


You'll be presented with a table of available tools and instructions for special commands. Interact with the AI assistant by typing your queries or commands.

### Special Commands

- `automode [number]`: Enter Autonomous mode with a specified number of iterations.
- `Ctrl+C`: While in automode, use this to exit and return to regular chat.
- `exit`: End the session and close the application.

## Configuration

The project uses environment variables for configuration. Create a `.env` file in the project root with the following structure:

python main.py


You'll be presented with a table of available tools and instructions for special commands. Interact with the AI assistant by typing your queries or commands.

### Special Commands

- `automode [number]`: Enter Autonomous mode with a specified number of iterations.
- `Ctrl+C`: While in automode, use this to exit and return to regular chat.
- `exit`: End the session and close the application.

## Configuration

The project uses environment variables for configuration. Create a `.env` file in the project root with the following structure:




ANTHROPIC_API_KEY=your_anthropic_api_key TAVILY_API_KEY=your_tavily_api_key


Replace `your_anthropic_api_key` and `your_tavily_api_key` with your actual API keys.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details on how to get involved.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Anthropic for the Claude AI model
- Tavily for the search API
- All the open-source libraries that made this project possible
