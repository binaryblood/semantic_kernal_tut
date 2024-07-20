import semantic_kernel as sk
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()
kernel = sk.Kernel()

kernel.add_text_completion_service("ollama", OpenAIChatCompletion(model_id="llama3-8b-8192", 
                                                                  api_key=os.getenv("GROQ_API_KEY"),
                                                                  endpoint="https://api.groq.com/openai/v1"))

print("A kernel is now ready.")

pluginsDirectory = "./plugins"

pluginBT = kernel.import_semantic_skill_from_directory(pluginsDirectory, "data_governance");

my_context = kernel.create_new_context()
my_context['term'] = 'Quality Assurance'

costefficiency_result = asyncio.run( kernel.run_async(pluginBT["definition_generator"], input_context=my_context))
costefficiency_str = str("### ✨ Suggestions for how to gain cost efficiencies\n" + str(costefficiency_result))
console.print(Markdown(costefficiency_str))