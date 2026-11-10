import os

from dotenv import load_dotenv
load_dotenv()

def main() -> None:
    print("Hello from langchain-course!")
    print(f"Anthropic API Key: {os.environ.get('ANTHROPIC_API_KEY')}")
