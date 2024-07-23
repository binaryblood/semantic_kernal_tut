##semantic functions vs native functions
* semantic functions are just to make prompts and get responses in natural language, they dont do logics
* native functions are function callig/tool calling in the code
* kernel.run_async() method accepts many functions to be chained one after the other as first param

##semantic plugins/functions
1. import from directory using 
```python
plugin_dg = kernel.import_semantic_skill_from_directory(pluginsDirectory, "data_governance");
```
2. Create from prompt using
```python
summary_function = kernel.create_semantic_function(prompt_template = sk_prompt,
                                                    description="Summarizes the input to length of an old tweet.",
                                                    max_tokens=200,
                                                    temperature=0.1,
                                                    top_p=0.5) 
```

##Native skills
1. Writing a class with functions that are specially annotated and are imported as skill
```python
from semantic_kernel.skill_definition import ( sk_function, sk_function_context_parameter,)
class ExoticLanguagePlugin:
    @sk_function(
        description="Takes text and converts it to pig latin",
        name="pig_latin",
        input_description="The text to convert to pig latin",
    )
    def pig_latin(self, sentence:str) -> str:
        #definition goes here
        pass
exotic_language_plugin = kernel.import_skill(ExoticLanguagePlugin(), skill_name="exotic_language_plugin")
pig_latin_function = exotic_language_plugin["pig_latin"]
final_result = asyncio.run( kernel.run_async(summary_function, pig_latin_function, input_str=sk_input) )
print(final_result)
```
|Ways to import Kernel functions| Description|
| :---------------- | :------: |
|kernel.import_semantic_skill_from_directory()| To import skills from directory (from config.json and sk_prompt.txt)|
|kernel.create_semantic_function()| To create a function from prompt as input string|
|kernel.import_skill()| To import a python function decorated with @sk_function as a plugin|