from collections import deque
from typing import List, Tuple, Optional
import pandas as pd
from itertools import combinations
import json



def convert_list_to_dict(lst):
    return {item: item for item in lst}

GRADES = ["good","brief","average","none"]
GRADES_OBJECT_PAIRS = convert_list_to_dict(GRADES)


def convert_data_grade_agent_supported(data, query: str=""):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data
        
    temp_query= ""
    for item in data:
        # if item['actions']:
        #     query += f"Actions: {', '.join(item['actions'])}\n"
        # else:
        #     query += "Actions: None\n"

        # if not item['discussions']:
        #     query += "Discussions: None\n"
        # else:
        #     query += f"Discussions: {', '.join(item['discussions'])}\n"


        # query += "Actions_and_Discussions:\n"
        
        # for action_discussion in item['actions_and_discussions']:
        #     if not action_discussion['actions']:
        #         query += "  Action: None,"
        #     else:
        #         query += f"  Action: {' '.join(action_discussion['actions'])},"

        #     if not action_discussion['discussions']:
        #         query += "  Discussion: None\n"
        #     else:
        #         query += f"  Discussion: {action_discussion['discussions']}\n"
        # query += "\n\n"
        temp_query += f"result: {item}\n"
        
    return query + temp_query

def get_combinations() -> List[Tuple[str, ...]]:
    combs = []
    products = GRADES
    
    print(f"products: {products} - get_combinations")

    for i in range(2, len(products) + 1):
        for c in combinations(products, i):
            combs.append(c)

    return combs


def merge_by_commodity(
    df_x,
    df_y,
    on: str = "Commodity",
) -> pd.DataFrame:
    if type(df_x) == str:
        df_x = pd.read_csv(df_x)
    if type(df_y) == str:
        df_y = pd.read_csv(df_y)
    df = pd.merge(df_x, df_y, on=on)
    return df
