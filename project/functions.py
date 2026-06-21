import pandas as pd
import json
from minsearch import Index

# Passing to the LLM
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Read data
def get_data(file_path="./courses_faq.parquet"):
    df = pd.read_parquet(file_path)
    return json.loads(df.to_json(orient="records"))

# Do search
def mini_search(document, text_fields=["question", "section", "answer"], filter=["course"]):
    index=Index(text_fields=text_fields, keyword_fields=filter)
    index.fit(document)
    return index


def search(index, question="", num_results=5, filter_dict={"course":"llm-zoomcamp"}):
    res = index.search(question, num_results=num_results, filter_dict=filter_dict)
    return res

# functions to build context from index search result.
def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()

# Generate Prompt
def build_prompt(question, res):
    base_prompt="""
Question:
{question}

Context:
{context}
    """

    context_formatted = build_context(res)
    base_prompt=base_prompt.format(question=question, context=context_formatted)
    return base_prompt

# Hit LLM and get result
def get_llm_response(sys_instruct, prompt):
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
    response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct),
            contents=prompt)
    
    return response