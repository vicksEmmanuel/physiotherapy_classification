import os
import json
import re
from typing import List, Dict, Callable
from time import sleep
import pandas as pd
import ollama

model_name = "llama3"

SUMMARY_PROMPT = (
    "You are a helpful physiotherapy expert studying a report of students' performance in a physiotherapy class. "
)
ANALYST_PROMPT = "You are a helpful physiotherapy expert scoring students' performance after they show steps and actions while analysing their patient."


def format_query(
    query: str,
    format_instruction: str = "You should format your response as a JSON object only and do not include any premise or suffix, just response with a json object format.",
):
    return f"{query}\n{format_instruction}"


def clean_json_string(json_string):
    # Remove newline characters and replace with space
    cleaned_string = re.sub(r'\n', ' ', json_string)
    # Remove extra whitespace
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    # Escape double quotes
    cleaned_string = cleaned_string.replace('"', '\\"')
    # Remove leading and trailing whitespace
    cleaned_string = cleaned_string.strip()
    return cleaned_string

def extract_json(response):
    if isinstance(response, dict) and 'message' in response:
        content = response['message']['content']
        cleaned_content = clean_json_string(content)
        try:
            json_content = json.loads(f'"{cleaned_content}"')
            return json_content
        except json.JSONDecodeError:
            print("Invalid JSON content")
            return None
    else:
        print("Unexpected response format")
        return None

def inference(
    query: str,
    system_content: str = ANALYST_PROMPT,
):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content":  query + "\njson"}
    ]
    
    try:
        response = ollama.chat(model=model_name, messages=messages, format="json")

        try:
            response =  extract_json(response.lower())
        except:
            response = response

        response = response['message']['content']
    except Exception as e:
        print(e)
        response = ""

    return response


def majority_voting_inference(
    query: str | List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    num_samples: int = 5,
    use_chain_of_thought: bool = False,
):
    responses = []
    for _ in range(num_samples):
        if use_chain_of_thought:
            response = chain_of_thought_inference(
                chain=query, system_content=system_content
            )["response"]
        else:
            response = inference(query, system_content)
        responses.append(response)

    performance_scores = [r["performance_score"] for r in responses if "performance_score" in r]
    if performance_scores:
        majority_score = max(set(performance_scores), key=performance_scores.count)
        response = {
            "performance_score": majority_score,
            "explanation": responses,
        }
    else:
        response = {
            "performance_score": None,
            "explanation": responses,
        }
    return response


def chain_of_thought_inference(
    chain: List[str | Callable],
    system_content: str = ANALYST_PROMPT,
):
    history = {}
    for query in chain:
        if isinstance(query, str):
            response = inference(query, system_content)
        else:
            previous_results = [history[k] for k in history.keys()]
            query = query(*previous_results)
            response = inference(query, system_content)
        history[query] = response

    return {
        "query": [{"prompt": q, "response": r} for q, r in history.items()],
        "response": response,
    }


def summarize(
    fname: str, grades: List[str]
) -> Dict[str, str]:
    grades = sorted(g.lower() for g in grades)
    summary_fname = fname.split(".")[0] + "-" + "-".join(grades) + ".json"
    if os.path.exists(summary_fname):
        return json.load(open(summary_fname))

    report = open(fname).read()
    query = f"Below is a report of students' performance in a physiotherapy class:\n\n{report}\n\n"

    format_instruction = f"""Please write a detailed summary of the report.

You should format your response as a JSON object. The JSON object should contain the following keys:
    overview: a string that describes, in detail, the overview of the report. Your summary should focus on factors that affect the overall performance of the students.
    """
    for g in grades:
        format_instruction += f"""
    {g}: a string that describes, in detail, information pertaining to {g} in the report. You should include information on {g} actions, discussions, and performance, as well as factors that affect them. 
        """
    query = format_query(query, format_instruction)
    response = inference(query, SUMMARY_PROMPT)
    try:
        response = json.loads(response.lower())
    except:
        response = response
    with open(summary_fname, "w") as f:
        json.dump(response, f, indent=4)
    return response