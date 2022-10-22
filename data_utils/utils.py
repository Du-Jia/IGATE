import os
from typing import *

import pandas as pd
import numpy as np
import json

def predictions2excel(file: str, tgt_dir:str =None) -> None:
    with open(file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f.readlines()]
    content = {'idx': [], 'pred': []}
    for line in lines:
        content['idx'].append(line['idx'])
        content['pred'].append(line['label'])
    return pd.DataFrame(content)
    
file = '/root/output/mrpc/case/p0-i0/eval_predictions_2022-06-15_00-48-15.jsonl'
cases = [
    '/root/output/mrpc/case/p0-i0/eval_predictions_2022-06-15_00-48-15.jsonl',
    '/root/output/mrpc/case/p0-i0/eval_predictions_2022-06-15_00-50-07.jsonl',
]

for i, case in enumerate(cases):
    casedf = predictions2excel(case)
    casedf.to_excel(f'case_{i}.xlsx', index=False)

