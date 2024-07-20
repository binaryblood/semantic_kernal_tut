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
plugin_dg = kernel.import_semantic_skill_from_directory(pluginsDirectory, "data_governance");


def generate_bt_definition(business_term):
    
    my_context = kernel.create_new_context()
    my_context['term'] = business_term
    generated_definition = asyncio.run( kernel.run_async(plugin_dg["definition_generator"], input_context=my_context))
    #costefficiency_str = str("### ✨ Term definition is\n" + str(costefficiency_result))
    #console.print(Markdown(costefficiency_str))
    return generated_definition

def classify_items(list_of_items, list_of_classes):
    my_context = kernel.create_new_context()
    my_context['items'] = list_of_items
    my_context['classes'] = list_of_classes
    generated_response = asyncio.run( kernel.run_async(plugin_dg["data_classification"], input_context=my_context))
    return generated_response

welcome_message = """Welcome to Data Quality & Governance platform!
      choose any of the below given options
      1. Generate Definition for business term
      2. Classify the data"""
choice = input(welcome_message)
choice = int(choice)
if(choice==1):
    # generate def
    term_name = input("Prove the business term")
    print(generate_bt_definition(term_name))
elif(choice==2):
    #classify data
    #example: Car, Ship, Bike, pizza, burger, apple, mango, orange
    list_of_items = input("Provide the list of items to classify seperated by comma")
    #example: vehicles, food, fruit, color
    list_of_classes = input("Provide the list of classes (seperated by comma) to which the provided items must be classified as")
    print(classify_items(list_of_items, list_of_classes))
else:
    print("Invlid option start again!")