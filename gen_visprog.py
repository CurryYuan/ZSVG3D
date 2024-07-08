import traceback
import re
import json
from tqdm import tqdm
import random
import jsonlines
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def decode_stimulus_string(s):
    """
    Split into scene_id, instance_label, # objects, target object id,
    distractors object id.

    :param s: the stimulus string
    """
    if len(s.split('-', maxsplit=4)) == 4:
        scene_id, instance_label, n_objects, target_id = \
            s.split('-', maxsplit=4)
        distractors_ids = ""
    else:
        scene_id, instance_label, n_objects, target_id, distractors_ids = \
            s.split('-', maxsplit=4)

    instance_label = instance_label.replace('_', ' ')
    n_objects = int(n_objects)
    target_id = int(target_id)
    distractors_ids = [int(i) for i in distractors_ids.split('-') if i != '']
    assert len(distractors_ids) == n_objects - 1

    return scene_id, instance_label, n_objects, target_id, distractors_ids


def is_explicitly_view_dependent(tokens):
    """
    :param df: pandas dataframe with "tokens" columns
    :return: a boolean mask
    """
    target_words = {
        'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost', 'looking', 'across'
    }
    return len(set(tokens).intersection(target_words)) > 0


def load_ref_data():
    anno_file = "data/nr3d.jsonl"
    scan_id_file = "data/scannet/splits/scannetv2_val.txt"
    split_scan_ids = set([x.strip() for x in open(scan_id_file, 'r')])

    ref_data = []
    with jsonlines.open(anno_file, 'r') as f:
        for item in f:
            if item['scan_id'] in split_scan_ids:
                # if (len(item['tokens']) > 24) and (not item['item_id'].startswith('scanrefer')):
                #     continue
                # if not is_explicitly_view_dependent(item['tokens']): continue
                # self.scan_ids.add(item['scan_id'])
                # self.scan_to_item_idxs[item['scan_id']].append(len(self.data))
                ref_data.append(item)

    return ref_data


if __name__ == '__main__':
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key="YOUR_API_KEY", temperature=0)

    ref_data = load_ref_data()
    random.shuffle(ref_data)
    ref_data = ref_data[:20]

    idx = 0
    new_data = []
    full_answers = []

    with open('prompts/prompt_n32.txt', 'r') as f:
        program_prompt = f.read()

    prompts = ChatPromptTemplate.from_messages([('system', program_prompt), ('human', '{input_prompt}')])

    pbar = tqdm(total=len(ref_data))

    while idx < len(ref_data):
        batch_idx = 0
        input_prompt = f"Please pares the following description to program according to the above examples and rules. Please follow the format of examples strictly.\n"
        while batch_idx < 10:
            ref = ref_data[idx]

            caption = ' '.join(ref['tokens'])
            input_prompt += f"[{idx}] Description: {caption}\n"

            batch_idx += 1
            idx += 1

            pbar.update(1)

        try:
            chain = prompts | llm | StrOutputParser()
            answers = chain.invoke({'input_prompt': input_prompt})
            # print(answers)
            full_answers.append(answers)

            answers = answers.split('\n\n')

            for answer in answers:
                answer = answer.split('\n')

                pattern = r'\[(\d+)\]'
                match = re.search(pattern, answer[0])

                if not match:
                    # print(answer[0])
                    continue
                ref = ref_data[int(match.group(1))]

                answer = '\n'.join(answer[1:])

                caption = ' '.join(ref['tokens'])

                hardness = decode_stimulus_string(ref['stimulus_id'])[2]
                easy_context_mask = hardness <= 2

                view_dep_mask = is_explicitly_view_dependent(ref['tokens'])

                new_data.append({
                    'scan_id': ref['scan_id'],
                    'target_id': ref['target_id'],
                    'caption': caption,
                    'program': answer,
                    'easy': easy_context_mask,
                    'view_dep': view_dep_mask
                })

        except Exception as e:
            traceback.print_exc()
            time.sleep(21)
            pass
        # break

    pbar.close()

    json.dump(new_data, open('data/visprog_example.json', 'w'), indent=4)
