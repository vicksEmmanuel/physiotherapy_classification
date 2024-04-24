import json
import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import List, Dict, Optional
import pandas as pd

from agent.agent import (
    PROJECT_ROOT,
    DeLLMaAgent,
    StateConfig,
    ActionConfig,
    PreferenceConfig,
)

sys.path.append(PROJECT_ROOT)
from utils.data_utils import GRADES, GRADES_OBJECT_PAIRS


class GradeAgent(DeLLMaAgent):
    grade: List[str] = GRADES
    grade_symbol_to_name_map: Dict[str, str] = GRADES_OBJECT_PAIRS
    system_content = "You are a physiotherapy teacher grading students' based on the actions and discussioned used to analyse their patients."

    states: Dict[str, Dict[str, str]] = {
        # product-agnostic state variables
        "agnostic": {
            "actions completeness": "the completeness of the actions like knee-examination, pelvis-check etc. used to analyse their patients",
            "discussions quality": "the quality of the discussions used to analyse their patients",
            "actions discussion alignment": "the alignment of the actions and discussions used to analyse their patients",
            "domain knowledge": "the domain knowledge of the students",
            "communication": "the communication of the students",
            "problem solving": "the problem solving of the students",
        },
        # product-specific state variables
        "specific": {
            "actions completeness": lambda c: f"the completeness of the actions like knee-examination, pelvis-check etc. used to analyse their patients",
            "discussions quality": lambda c: f"the quality of the discussions used to analyse their patients",
            "actions discussion alignment": lambda c: f"the alignment of the actions and discussions used to analyse their patients",
            "domain knowledge": lambda c: f"the domain knowledge of the students",
            "communication": lambda c: f"the communication of the students",
            "problem solving": lambda c: f"the problem solving of the students",
        },
    }

    product = "grade"

    def __init__(
        self,
        choices: List[str],
        path: str = os.path.join(PROJECT_ROOT, "data/grades/"),
        raw_context_fname: str = "report.json",
        temperature: float = 0.0,
        state_config: Optional[dataclass] = None,
        action_config: Optional[dataclass] = None,
        preference_config: Optional[dataclass] = None,
        agent_name: str = "phsiotherapy_teacher",
    ):
        assert set(choices).issubset(set(self.grade))
        self.choices = sorted(set(choices))
        self.path = path
        utility_prompt = f"I'm a physiotherapy teacher grading students' based on the actions and discussioned used to analyse their patients."

        super().__init__(
            path,
            raw_context_fname,
            temperature,
            utility_prompt,
            state_config,
            action_config,
            preference_config,
            agent_name,
        )

        if (
            self.state_config.state_enum_mode != "base"
            and len(self.state_config.states) == 0
        ):
            self.state_config.states = self._format_state_dict()

    def _format_state_dict(self):
        state2desc = self.states["agnostic"].copy()
        for choice, variable in product(
            self.choices, sorted(self.states["specific"].keys())
        ):
            state2desc[
                f"{variable} {self.grade_symbol_to_name_map[choice]} ({choice.upper()})".lower()
            ] = self.states["specific"][variable]
        return state2desc
    
    def _format_grade_context(self, grade: str):
        with open(os.path.join(self.path, "report.json"), 'r') as file:
            data = json.load(file)

        if grade in data:
            grade_data = data[grade]
            query = f"Below are the information about grade {grade} (i.e. {self.grade_symbol_to_name_map[grade]}).\n\n"
            
            for item in grade_data:
                query += f"Actions: {', '.join(item['actions'])}\n"
                query += f"Discussions: {', '.join(item['discussions'])}\n"
                query += "Actions and Discussions:\n"
                for action_discussion in item['actions_and_discussions']:
                    query += f"  Action: {action_discussion['actions']}\n"
                    query += f"  Discussion: {action_discussion['discussions']}\n"
                query += "\n"
            
            return query
        else:
            return f"No data found for grade {grade}."
    
    def prepare_context(self) -> str:
        context = f"""Below are the grades I am considering: {", ".join(self.choices)}. I would like to know which grade I should give based on the actions and discussions used to analyse their patients.
        I can only choose one grade.
        """
        for p in self.choices:
            context += self._format_grade_context(p)
        return context



if __name__ == "__main__":
    # # Example to produce the belief distribution prompt
    agent = GradeAgent(
        choices=["good", "brief", "average"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(),
        preference_config=PreferenceConfig(),
    )
    belief_distribution_prompt = agent.prepare_belief_dist_generation_prompt()

    # Example to produce the full dellma prompt
    agent = GradeAgent(
        choices=["good", "brief", "average"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(),
        preference_config=PreferenceConfig(pref_enum_mode="order", sample_size=50),
    )
    dellma_prompt = agent.prepare_dellma_prompt()
    print(dellma_prompt)
