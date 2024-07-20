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
#model_id="llama3-8b-8192",
model_id="gemma2-9b-it"
kernel.add_text_completion_service("ollama", OpenAIChatCompletion(model_id=model_id,
                                                                  api_key=os.getenv("GROQ_API_KEY"),
                                                                  endpoint="https://api.groq.com/openai/v1"))

print("A kernel is now ready.")

pluginsDirectory = "./plugins"
'''
plugin_dg = kernel.import_semantic_skill_from_directory(pluginsDirectory, "data_governance");

from semantic_kernel.planning import ActionPlanner

planner = ActionPlanner(kernel)

from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill
kernel.import_skill(MathSkill(), "math")
kernel.import_skill(FileIOSkill(), "fileIO")
kernel.import_skill(TimeSkill(), "time")
kernel.import_skill(TextSkill(), "text")

print("Adding the tools for the kernel to do math, to read/write files, to tell the time, and to play with text.")

ask = "What is the sum of 110 and 990?"

print(f"🧲 Finding the most similar function available to get that done...")
plan = asyncio.run( planner.create_plan_async(goal=ask))
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")
'''
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.core_skills.text_skill import TextSkill
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig

writer_plugin = kernel.import_semantic_skill_from_directory(pluginsDirectory, "LiterateFriend")

# create an instance of sequential planner, and exclude the TextSkill from the list of functions that it can use.
# (excluding functions that ActionPlanner imports to the kernel instance above - it uses 'this' as skillName)
planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))

ask = """
Tomorrow is Valentine's day. I need to come up with a poem. Translate the poem to French.
"""

plan = asyncio.run( planner.create_plan_async(goal=ask))
result = asyncio.run(  plan.invoke_async())

for index, step in enumerate(plan._steps):
    print(f"✅ Step {index+1} used function `{step._function.name}`")

trace_resultp = True

if trace_resultp:
    print("Longform trace:\n")
    for index, step in enumerate(plan._steps):
        print("Step:", index+1)
        print("Description:",step.description)
        print("Function:", step.skill_name + "." + step._function.name)
        print("Input vars:", step._parameters._variables)
        print("Output vars:", step._outputs)
        if len(step._outputs) > 0:
            print( "  Output:\n", str.replace(result[step._outputs[0]],"\n", "\n  "))

print(f"## ✨ Generated result from the ask: {ask}\n\n---\n" + str(result))
