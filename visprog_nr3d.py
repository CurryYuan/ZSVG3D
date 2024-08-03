import sys
import time
from icecream import ic
import json
import traceback
import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import argparse

from zsvg.program import ProgramInterpreter
from zsvg.util import calc_iou, create_logger
from zsvg.loc_interpreters import load_pc

if __name__ == '__main__':
    # add an argument
    parser = argparse.ArgumentParser(description='visprog nr3d.')
    parser.add_argument('--prog_path', type=str, default='data/nr3d_val.json', help='exp name')
    parser.add_argument('--exp_name', type=str, default='test', help='exp name')
    args = parser.parse_args()

    with open(args.prog_path, 'r') as f:
        programs = json.load(f)

    interpreter = ProgramInterpreter()

    correct_25 = 0
    correct_50 = 0
    correct_easy = 0
    correct_dep = 0
    easy_total = 0
    dep_total = 0
    recall = 0

    for program in tqdm(programs):
        prog_str = program['program']
        scan_id = program['scan_id']
        caption = program['caption']

        obj_ids, inst_locs, center, _ = load_pc(scan_id)

        index = obj_ids.index(program['target_id'])
        target_box = inst_locs[index]

        init_state = {'scan_id': scan_id}

        if program['easy']:
            easy_total += 1
        if program['view_dep']:
            dep_total += 1

        try:
            result, prog_state = interpreter.execute(prog_str, init_state=init_state, inspect=False)

            for i, target in enumerate(prog_state['BOX0']):
                pred_box = target['obj_loc']
                best_iou = calc_iou(pred_box, target_box)
                target['iou'] = best_iou
                if best_iou > 0.25:
                    recall += 1
                    break

            pred_box = result[0]['obj_loc']
            iou = calc_iou(pred_box, target_box)

            if iou >= 0.25:
                correct_25 += 1
                # if iou >= 0.5:
                #     correct_50 += 1
                if program['easy']:
                    correct_easy += 1
                if program['view_dep']:
                    correct_dep += 1

        except Exception as e:
            pass

    logger = create_logger(args.exp_name)

    logger.info('Easy {} {} / {}'.format(correct_easy / easy_total, correct_easy, easy_total))
    logger.info('Hard {} {} / {}'.format((correct_25 - correct_easy) / (len(programs) - easy_total),
                                         correct_25 - correct_easy,
                                         len(programs) - easy_total))
    logger.info('View-Dep {} {} / {}'.format(correct_dep / dep_total, correct_dep, dep_total))
    logger.info('View-Indep {} {} / {}'.format((correct_25 - correct_dep) / (len(programs) - dep_total),
                                               correct_25 - correct_dep,
                                               len(programs) - dep_total))
    logger.info('Acc@25 {} {} / {}'.format(correct_25 / len(programs), correct_25, len(programs)))
    # print('Acc@50', correct_50 / len(programs), correct_50, '/', len(programs))
    logger.info('Recall {} {} / {}'.format(recall / len(programs), recall, len(programs)))
