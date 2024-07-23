import asyncio

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from dotenv import load_dotenv
import os

load_dotenv()
from semantic_kernel.functions import kernel_function
from typing import List

class EmailPlugin:
    @kernel_function(
        name="send_email",
        description="Sends an email to a recipient."
    )
    async def send_email(self, recipient_emails: str|List[str], subject: str, body: str):
        # Add logic to send an email using the recipient_emails, subject, and body
        # For now, we'll just print out a success message to the console
        print(f"Email sent! as recepient:{recipient_emails}, subject:{subject}, body:{body}")

async def main():
    # Initialize the kernel
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(
        ai_model_id = "llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    ))

    kernel.add_text_completion_service("Groq", OpenAIChatCompletion(model_id="llama3-8b-8192", 
                                                                  api_key=os.getenv("GROQ_API_KEY"),
                                                                  endpoint="https://api.groq.com/openai/v1"))
    

    # Add a plugin (the EmailPlugin class is defined above)
    kernel.add_plugin(
        EmailPlugin(),
        plugin_name="Email",
    )

    chat_completion : OpenAIChatCompletion = kernel.get_service(type=ChatCompletionClientBase)

    # Enable planning
    execution_settings = OpenAIChatPromptExecutionSettings(tool_choice="auto")
    execution_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={})

    # Create a history of the conversation
    history = ChatHistory()

    # Start the conversation
    while True:
        # Get user input
        user_input = input("User > ")
        history.add_user_message(user_input)

        # Get the response from the AI
        result = (await chat_completion.get_chat_message_contents(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
            arguments=KernelArguments(),
        ))[0]

        # Print the response
        print("Assistant > " + str(result))
        history.add_assistant_message(str(result))

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())