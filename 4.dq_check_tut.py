import semantic_kernel as sk
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()
kernel = sk.Kernel()
#model_id="llama3-8b-8192",
model_id="gemma2-9b-it"
kernel.add_text_completion_service("groq", OpenAIChatCompletion(model_id=model_id,
                                                                  api_key=os.getenv("GROQ_API_KEY"),
                                                                  endpoint="https://api.groq.com/openai/v1"))

print("A kernel is now ready.")

pluginsDirectory = "./plugins"

from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig
#from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
import json

from semantic_kernel.skill_definition import (sk_function,)
writer_plugin = kernel.import_semantic_skill_from_directory(pluginsDirectory, "data_quality")
class EmailDataQualityChecker:
    
    @sk_function(
        description="Takes email text and checks for the quality",
        name="check_email_quality",
        input_description="list of emails comma seperated",
    )
    def check_email_quality(self, emails:str) -> str:
        all_emails = emails.split(',')
        qa_checked = []
        for email in all_emails:
            trimmed_mail = email.trim()
            qa = "fail"
            if(trimmed_mail == trimmed_mail.upper()):
                qa="pass"
            qa_checked.append({"mail": trimmed_mail, "quality": qa})
        return json.dumps(qa_checked)

dq_plugin = kernel.import_skill(EmailDataQualityChecker(), skill_name="Email Data Quality Checker")
check_email_quality = dq_plugin["check_email_quality"]
# create an instance of sequential planner, and exclude the TextSkill from the list of functions that it can use.
# (excluding functions that ActionPlanner imports to the kernel instance above - it uses 'this' as skillName)
planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))

ask = """
I want the email ids from the below content.
The content is:
hi Mr Stark <tony.stark@gmail.com>, this is MR.Howard <HOWARD.STARK@GMAIL.COM>,
I am delighted to tell you that we can use this semantic kernal to write the software for the Iron Man suite
I have informed Mr.PeterParker <peter.parker@spidymail.net> about the information too.
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
