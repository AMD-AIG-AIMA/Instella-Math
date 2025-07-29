# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import ast
import numpy as np
from itertools import permutations

def parse_answer(answer: str, delimiter: str = None):
    if not answer:
        return None
    rows = answer.split("\n")
    result = []
    
    for row in rows:
        row = row.strip()
        if not row:
            continue
        
        columns = row.split(delimiter) if delimiter else row.split()
        result.append([int(grid) if grid.strip().isdigit() else None for grid in columns])
    
    return result if all(None not in row for row in result) else None

def parse_list_str(input_string):
    try:
        return ast.literal_eval(input_string)
    except (ValueError, SyntaxError):
        return None

def preprocess_answer_to_matrix(raw_answer):
    if isinstance(raw_answer, list):
        return raw_answer
    
    answer = parse_answer(raw_answer, delimiter=' ')
    if answer is None:
        answer = parse_answer(raw_answer, delimiter=',')
    if answer is None:
        answer = parse_list_str(raw_answer)
    return answer

def extract_last_code_block(text: str):
    code_blocks = re.findall(r'```.*?\n(.*?)```', text, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    return code_blocks[-1].strip() if code_blocks else None

def get_all_permutations(arr):
  perms = list(permutations(arr))
  
def get_cage_scores(grid, cages):
    scores=[]
    for cage in cages:
        score=0
        cells = cage['cells']
        operation= cage['operation']
        if operation == '+':
            sum_pred = 0
            for cell in cells:
                sum_pred+=grid[cell[0], cell[1]]
            if sum_pred==cage['target']:
                score=1
        if operation == '*':
            sum_pred = 0
            for cell in cells:
                sum_pred*=grid[cell[0], cell[1]]
            if sum_pred==cage['target']:
                score=1
        if operation == '=':
            if len(cells)==1:
                if grid[cells[0][0], cells[0][1]]==cage['target']:
                    score=1
        if operation == '/':
            cell_values =[]
            for cell in cells:
                cell_values.append(grid[cell[0], cell[1]])
            candidate_orders=list(permutations(cell_values))
            for candidate in candidate_orders:
                div_pred = list(candidate)[0]
                for c in list(candidate)[1:]:
                    div_pred/=c
                if int(div_pred)==cage['target']:
                    score=1
        if operation == '-':
            cell_values =[]
            for cell in cells:
                cell_values.append(grid[cell[0], cell[1]])
            candidate_orders=list(permutations(cell_values))
            for candidate in candidate_orders:
                sub_pred = list(candidate)[0]
                for c in list(candidate)[1:]:
                    sub_pred-=c
                if int(sub_pred)==cage['target']:
                    score=1
        scores.append(score)
    return scores
def compute_score(solution_str, ground_truth, cages, score=1.):
    extracted_response = extract_last_code_block(solution_str)
    if not extracted_response:
        return 0, [0]*len(cages)
    
    normalized_pred = preprocess_answer_to_matrix(extracted_response)
    normalized_gt = preprocess_answer_to_matrix(ground_truth)
    
    if normalized_pred is None:
        return 0, [0]*len(cages)
    if np.array(normalized_pred).shape != np.array(normalized_gt).shape:
        return 0, [0]*len(cages)
    try:
        cage_scores = get_cage_scores(np.array(normalized_pred), cages)
        if (np.array(normalized_pred) == np.array(normalized_gt)).all():
            return 1, cage_scores
        else:
            return 0, cage_scores
    except Exception as ex:
        return 0, [0]*len(cages)