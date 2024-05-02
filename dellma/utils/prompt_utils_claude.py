# 

import os
import json
from typing import List, Dict, Callable
from time import sleep
import anthropic
import pandas as pd

from openai import OpenAI

api_key =  os.environ["ANTHROPIC_API_KEY"]
CLIENT = anthropic.Anthropic(api_key=api_key)

SUMMARY_PROMPT = (
    "You are a helpful physiotherapy expert studying a report of students' performance in a physiotherapy class. "
)
ANALYST_PROMPT = "You are a helpful physiotherapy expert scoring students' performance after they show steps and actions while analysing their patient."


def format_query(
    query: str,
    format_instruction: str = "You should format your response as a JSON object only and do not include any premise or suffix, just response with a json object format.",
):
    return f"{query}\n{format_instruction}"


def inference(
    query: str,
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.0,
):
    success = False
    while not success:
        try:
            response = CLIENT.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": query },
                ],
                system=system_content,
                temperature=temperature,
            )


            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        response_json = response.model_dump_json(indent=4)
        response_dump = json.loads(response_json)
        response_content = response_dump["content"]

        if isinstance(response_content, list):
            response = response_content[0].get("text", "")
        else:
            response = ""

        # Replace every word json with empty string
        response = response.replace("json", "")
        # Replace every ``` with empty string
        response = response.replace("```", "")

    except Exception as e:
        response = ""

    try:
        response = json.loads(response.lower())
    except:
        response = response

    print(f"response: {response}")

    return response


def majority_voting_inference(
    query: str | List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.7,
    num_samples: int = 5,
    use_chain_of_thought: bool = False,
):
    responses = []
    for _ in range(num_samples):
        if use_chain_of_thought:
            response = chain_of_thought_inference(
                chain=query, system_content=system_content, temperature=temperature
            )["response"]
        else:
            response = inference(query, system_content, temperature)
        responses.append(response)

    print(f"responses:  ======>>> \n\n\n\n{responses}")

    decisions = []

    for r in responses:
        try:
            decisions.append(r['decision'])
        except Exception as e:
            print(e)

    print(f" decision: {decisions}")

    majority_decision = max(set(decisions), key=decisions.count)

    response = {
        "decision": majority_decision,
        "explanation": responses,
    }
    return response


def chain_of_thought_inference(
    chain: List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.5,
):
    history = {}
    for query in chain:
        if isinstance(query, str):
            response = inference(query, system_content, temperature)
        else:
            previous_results = [history[k] for k in history.keys()]
            query = query(*previous_results)
            response = inference(query, system_content, temperature)
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