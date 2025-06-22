# Import necessary modules
import getpass  # For securely getting API keys from user input
import os  # For handling environment variables and file operations

try:
    # Load environment variables from a .env file (requires python-dotenv package)
    # This allows us to store configuration in an external file
    from dotenv import load_dotenv
    
    # Actually load the environment variables from the .env file
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, just pass without error
    pass

# Set tracing for LangSmith (useful for debugging and monitoring)
os.environ["LANGSMITH_TRACING"] = "true"

# Check if LANGSMITH_API_KEY exists in environment variables
# If not, prompt user to enter it securely
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )

# Check if LANGSMITH_PROJECT exists in environment variables
# If not, prompt user to enter project name with a default value
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    
    # If no project name was entered, set default value
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

# Check for Azure OpenAI API key in environment variables
# If not found, prompt user to enter it securely
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

# Import the AzureChatOpenAI class from langchain_openai module
from langchain_openai import AzureChatOpenAI

# Initialize the Azure OpenAI model with configuration from environment variables
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  # Azure endpoint URL
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],  # Name of deployed model
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"]  # API version to use
)

# The following lines are commented out examples of different ways to invoke the model:
"""
# Example 1: Using system and human messages directly
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

model.invoke(messages)  # Invoke the model with message list

# Example 2: Passing a simple string input
model.invoke("Hello")  # Direct text input to the model

# Example 3: Using dictionary format for messages
model.invoke([{"role": "user", "content": "Hello"}])

# Example 4: Using HumanMessage objects directly
model.invoke([HumanMessage("Hello")])
"""

# Import ChatPromptTemplate from langchain_core.prompts module
from langchain_core.prompts import ChatPromptTemplate

# Define system message template for translation task
system_template = "Translate the following from English into {language}"

# Create a prompt template with system and user messages
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Invoke the prompt template with specific parameters
prompt = prompt_template.invoke({"language": "Chinese", "text": "hi!"})

# Convert prompt to message format for model invocation
prompt.to_messages()

# Get response from the model using the prepared prompt
response = model.invoke(prompt)
print(response.content)  # Print the model's response content
