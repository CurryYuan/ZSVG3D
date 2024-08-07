import pandas as pd
import numpy as np
import random
import json
import pickle
from tqdm import tqdm
import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocess.utils import get_train_val_split, load_pc
from models.pcd_classifier import PcdClassifier



if __name__ == '__main__':

    train_scene_ids, val_scene_ids = get_train_val_split()
    scene_ids = train_scene_ids + val_scene_ids

    model = PcdClassifier().cuda()
    ckpt_path = 'weights/pnext_cls.pth'

    weights = torch.load(ckpt_path, map_location='cpu')
    info = model.load_state_dict(weights, strict=False)
    print(info)

    # save model weights
    # torch.save(model.state_dict(), 'weights/pnext_cls.pth')

    model.eval()

    data = {}

    for scan_id in tqdm(scene_ids):
        batch_labels, obj_ids, inst_locs, center, batch_pcds = load_pc(scan_id, keep_background=False)

        # obj_ids = list(range(len(batch_labels)))

        obj_embeds = model(batch_pcds[..., :4].cuda())     # (B, D)
        obj_embeds = torch.nn.functional.normalize(obj_embeds, p=2, dim=-1)

        data[scan_id] = {
            'batch_labels': batch_labels,
            'obj_ids': obj_ids,
            'inst_locs': inst_locs,
            'center': center,
            'obj_embeds': obj_embeds.detach().cpu()
        }

        # break

    # save in pickle
    with open('data/scannet/feats_3d.pkl', 'wb') as f:
        pickle.dump(data, f)
