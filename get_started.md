# Getting Started

Follow these steps to set up and run the AI Assistant project.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:



git clone https://github.com/yourusername/ai-assistant.git cd ai-assistant


2. Create a virtual environment:



python -m venv venv source venv/bin/activate # On Windows, use venv\Scripts\activate


3. Install the required packages:



pip install -r requirements.txt


4. Set up your environment variables:
Create a `.env` file in the project root and add your API keys:



ANTHROPIC_API_KEY=your_anthropic_api_key TAVILY_API_KEY=your_tavily_api_key


## Usage

Run the main script:



python main.py


Follow the on-screen instructions to interact with the AI assistant.

### Special Commands

- Type 'automode [number]' to enter Autonomous mode with a specific number of iterations.
- While in automode, press Ctrl+C at any time to exit and return to regular chat.
- Type 'exit' to end the session.

## Troubleshooting

If you encounter any issues, please check the following:

1. Ensure all required environment variables are set correctly in the `.env` file.
2. Verify that you're using a compatible Python version.
3. Make sure all dependencies are installed correctly.

If problems persist, please open an issue on the GitHub repository.



