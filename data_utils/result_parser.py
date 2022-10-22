import os
from typing import Dict,List

import pandas as pd
import numpy as np


def _get_params(params: str) -> Dict:
    params_dic = {}
    params = params.strip()[:-4].split(':')[-1].split(',')

    for param in params:
        key, value = param.split('=')
        key = key.strip()
        params_dic[key] = float(value) if key == "LR" else int(value)

    return params_dic


def _get_metric(record: str, metric='acc') -> Dict:
    value = float(record.split(':')[-1].split('+-')[0].strip())
    return value


def _result_collector(abspath: str, record_cnt: int, experiments=["max", "none"]) -> Dict[str, List]:
    records = {
        'INFO_TYPE': [],
        'SEED': [],
        'LR': [],
        'BS': [],
        'PL': [],
        'acc': []
    }

    for info_type in experiments:
        result_file = os.path.join(abspath, f'{info_type}/result_test.txt')
        # result_file = os.path.join(abspath, f'{info_type}/result_adaptive.txt')
        with open(result_file, 'r') as f:
            raw_records = f.readlines()

        for index, raw_record in enumerate(raw_records):
            if index % 7 == 0:
                params = _get_params(raw_record)
                records['INFO_TYPE'].append(info_type)
                [records[key].append(value) for key, value in params.items()]
                records['acc'].append(_get_metric(raw_records[index + 2]))

    return pd.DataFrame(records)


def result_parser(records: pd.DataFrame) -> Dict:
    # for seed in (10, 12, 21, 42, 87):
    #     for info_type in ('max', 'none'):
    #         tmp = records[records['SEED'] == seed]
    #         tmp = tmp[tmp['INFO_TYPE'] == info_type]
    #         print(tmp)
    info = records.groupby(['SEED', 'INFO_TYPE', 'PL'])['acc'].max()

    print(info)
    info.to_excel('qqp_pl4_v1.xlsx')


if __name__ == "__main__":
    abspath = '/home/users/dujia/NLP/p-tuning/few_out/qqp'
    records = _result_collector(abspath, record_cnt=4)
    # records.to_excel('exp1.xlsx', index=False)
    result_parser(records)