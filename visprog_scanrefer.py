import json
import traceback
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

from zsvg.program import ProgramInterpreter
from zsvg.util import calc_iou, create_logger
from preprocess.utils import load_pc

if __name__ == '__main__':
    # add an argument
    parser = argparse.ArgumentParser(description='visprog scanrefer.')
    parser.add_argument('--prog_path', type=str, default='data/scanrefer_val.json', help='exp name')
    parser.add_argument('--exp_name', type=str, default='test', help='exp name')
    args = parser.parse_args()

    with open(args.prog_path, 'r') as f:
        programs = json.load(f)

    # Load label map
    label_map_file = 'data/scannetv2-labels.combined.tsv'
    labels_pd = pd.read_csv(label_map_file, sep='\t', header=0)

    interpreter = ProgramInterpreter(loc='LOC_3D_pred')

    correct_25 = 0
    correct_50 = 0
    unique_25 = 0
    unique_50 = 0
    unique_total = 0
    total = 0
    recall = 0

    success_programs = []

    for program in tqdm(programs):
        prog_str = program['program']
        scan_id = program['scan_id']

        batch_labels, obj_ids, inst_locs, center, batch_pcds = load_pc(scan_id)
        batch_class_ids = []

        for obj_label in batch_labels:
            label_ids = labels_pd[labels_pd['raw_category'] == obj_label]['nyu40id']
            label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
            batch_class_ids.append(label_id)

        index = obj_ids.index(program['target_id'])
        target_box = inst_locs[index]
        target_class_id = batch_class_ids[index]

        unique = (np.array(batch_class_ids) == target_class_id).sum() == 1

        if unique:
            unique_total += 1

        init_state = {'scan_id': scan_id}

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

                success_programs.append(program)

                if unique:
                    unique_25 += 1
                if iou >= 0.5:
                    correct_50 += 1
                    if unique:
                        unique_50 += 1

        except Exception as e:
            pass

    logger = create_logger(args.exp_name)

    logger.info('Unique@25 {} {} / {}'.format(unique_25 / unique_total, unique_25, unique_total))
    logger.info('Unique@50 {} {} / {}'.format(unique_50 / unique_total, unique_50, unique_total))
    logger.info('Multiple@25 {} {} / {}'.format((correct_25 - unique_25) / (len(programs) - unique_total),
                                                correct_25 - unique_25,
                                                len(programs) - unique_total))
    logger.info('Multiple@50 {} {} / {}'.format((correct_50 - unique_50) / (len(programs) - unique_total),
                                                correct_50 - unique_50,
                                                len(programs) - unique_total))
    logger.info('Acc@25 {} {} / {}'.format(correct_25 / len(programs), correct_25, len(programs)))
    logger.info('Acc@50 {} {} / {}'.format(correct_50 / len(programs), correct_50, len(programs)))
    logger.info('Recall {} {} / {}'.format(recall / len(programs), recall, len(programs)))
