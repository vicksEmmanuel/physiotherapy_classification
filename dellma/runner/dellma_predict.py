import os
import json
import re
from typing import Dict
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from dellma.agent.physiotherapyagent import GradeAgent
# from dellma.utils.prompt_utils_gpt import (
#     inference,
#     majority_voting_inference,
#     chain_of_thought_inference,
# )

from dellma.utils.prompt_utils_claude import (
    inference,
    majority_voting_inference,
    chain_of_thought_inference,
)

# from dellma.utils.prompt_utils_llama import (
#     inference,
#     majority_voting_inference,
#     chain_of_thought_inference,
# )

from dellma.utils.data_utils import convert_data_grade_agent_supported, get_combinations, GRADES
from dellma.agent.agent import StateConfig, ActionConfig, PreferenceConfig
from functools import partial


def parse_baseline_response(response: Dict[str, str]) -> int:
    try:
        decision = int(response["decision"].split(".")[0].split()[1]) - 1
    except ValueError:
        decision = -1
    return decision

def process_grades(
    sc_samples: int = 5,
    dellma_mode: str = "zero-shot",
    current_physiotherapy_analysis_to_grade: str = "",
    sample_size: int = 64,
    minibatch_size: int = 32,
    overlap_pct: float = 0.25,
):
    agent_init_fct = partial(
        GradeAgent,
        current_physiotherapy_analysis_to_grade=current_physiotherapy_analysis_to_grade,
    )

    action_config = ActionConfig()

    if dellma_mode.startswith("rank"):
        state_enum_mode = "sequential"
        preference_config = PreferenceConfig(
            pref_enum_mode=dellma_mode,
            sample_size=sample_size,
            # if dellma_mode is rank, then all below are ignored
            minibatch_size=minibatch_size,
            overlap_pct=overlap_pct,
        )
    elif dellma_mode in ["zero-shot", "self-consistency", "cot"]:
        state_enum_mode = "base"
        preference_config = PreferenceConfig()
    else:
        raise ValueError(f"Unknown dellma mode: {dellma_mode}")

    combs = get_combinations()
    choices = combs[-1]

    agent = agent_init_fct(
        choices=choices,
        state_config=StateConfig(state_enum_mode),
        action_config=action_config,
        preference_config=preference_config,
    )
    if dellma_mode == "cot":
        prompts = agent.prepare_chain_of_thought_prompt()
    else:
        prompts = agent.prepare_dellma_prompt()

    if type(prompts) == str:
        prompts = [prompts]
    if dellma_mode == "cot":
        inference_fct = partial(
            chain_of_thought_inference,
            system_content=agent.system_content,
        )
    elif dellma_mode == "self-consistency":
        inference_fct = partial(
            majority_voting_inference,
            system_content=agent.system_content,
            num_samples=sc_samples,
        )
    else:
        inference_fct = partial(
            inference,
            system_content=agent.system_content,
        )

    # TODO: For all LLMS
    if dellma_mode == "cot":
        output = inference_fct(chain=prompts)
        response = output["response"]
        prompt = output["query"]

        # Regular expression pattern to match JSON-like strings
        json_pattern = re.compile(r'{.*}', re.DOTALL)

        if isinstance(response, dict):
            match = response
            try:
                decision = match.get("decision")
                if decision:
                    print("Decision:", decision)
                else:
                    print("Decision key not found in the JSON object.")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
        else:
            # Find the first match of the JSON pattern in the text
            match = json_pattern.search(response)
            if match:
                try:
                    json_obj = json.loads(match.group())
                    decision = json_obj.get("decision")
                    if decision:
                        print("Decision:", decision)
                    else:
                        print("Decision key not found in the JSON object.")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {str(e)}")
            else:
                print("No JSON-like string found in the text.")

        

        return prompt, response
                
    else:
        for i, prompt in enumerate(prompts):
            # save dellma prompt
            response = inference_fct(prompt)
            # save dellma response
            return prompt, response


if __name__ == "__main__":
    sc_samples = 5
    current_student_action = '''
            [{
                "actions": [],
                "discussions": [
                    " Looking at the front of the patients, just looking at the posture, generally, any swelling",
                    " and the knee, the muscle bulk, any potential injuries.",
                    " The side, just looking at the pelvis, the rear, just looking at the back and the spine,",
                    "",
                    " and that's it, I think I'll go on.",
                    ""
                ],
                "actions_and_discussions": [
                    {
                        "actions": [],
                        "discussions": " Looking at the front of the patients, just looking at the posture, generally, any swelling",
                        "start_time": 0,
                        "end_time": 15.0
                    },
                    {
                        "actions": [],
                        "discussions": " and the knee, the muscle bulk, any potential injuries.",
                        "start_time": 15.0,
                        "end_time": 26.0
                    },
                    {
                        "actions": [],
                        "discussions": " The side, just looking at the pelvis, the rear, just looking at the back and the spine,",
                        "start_time": 25.0,
                        "end_time": 42.0
                    },
                    {
                        "actions": [],
                        "discussions": "",
                        "start_time": 42.0,
                        "end_time": 44.0
                    },
                    {
                        "actions": [],
                        "discussions": " and that's it, I think I'll go on.",
                        "start_time": 44.0,
                        "end_time": 48.0
                    },
                    { "actions": [], "discussions": "", "start_time": 48.0, "end_time": 73 }
                ]
            }]
        '''

    # query, result = process_grades(
    #     sc_samples=500,
    #     dellma_mode="zero-shot",
    #     current_physiotherapy_analysis_to_grade=convert_data_grade_agent_supported(current_student_action, query="")
    # )

    # print(f"Value zero-shot: {result}")


    # query, result = process_grades(
    #     sc_samples=5,
    #     dellma_mode="cot",
    #     current_physiotherapy_analysis_to_grade=convert_data_grade_agent_supported(current_student_action, query="")
    # )

    # print(f"Value cot: {result}")

    query, result = process_grades(
        sc_samples=sc_samples,
        dellma_mode="self-consistency",
        current_physiotherapy_analysis_to_grade=convert_data_grade_agent_supported(current_student_action, query="")
    )

    print(f"Value self-consistency: {result}")

    


    