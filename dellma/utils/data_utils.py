from collections import deque
from typing import List, Tuple, Optional
import pandas as pd
from itertools import combinations



def convert_list_to_dict(lst):
    return {item: item for item in lst}

GRADES = ["good","brief","average"]
GRADES_OBJECT_PAIRS = convert_list_to_dict(GRADES)




def get_combinations() -> List[Tuple[str, ...]]:
    combs = []
    products = GRADES
    
    print(f"products: {products} - get_combinations")

    for i in range(2, len(products) + 1):
        for c in combinations(products, i):
            combs.append(c)

    return combs


def merge_by_commodity(
    df_x: pd.DataFrame | str,
    df_y: pd.DataFrame | str,
    on: str = "Commodity",
) -> pd.DataFrame:
    if type(df_x) == str:
        df_x = pd.read_csv(df_x)
    if type(df_y) == str:
        df_y = pd.read_csv(df_y)
    df = pd.merge(df_x, df_y, on=on)
    return df
