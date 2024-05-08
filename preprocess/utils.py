


import json
import os
import numpy as np
import torch


def get_train_val_split():
    train_split_path = "data/scannet/splits/scannetv2_train.txt"
    val_split_path = "data/scannet/splits/scannetv2_val.txt"
    train_scene_ids = []
    val_scene_ids = []

    with open(train_split_path, "r") as f:
        for line in f.readlines():
            train_scene_ids.append(line.strip())

    with open(val_split_path, "r") as f:
        for line in f.readlines():
            val_scene_ids.append(line.strip())

    return train_scene_ids, val_scene_ids


def load_pc(scan_id, keep_background = False, scan_dir = '/221019046/Projects/vil3dref/datasets/referit3d/scan_data'):
    pcds, colors, _, instance_labels = torch.load(
        os.path.join(scan_dir, 'pcd_with_global_alignment', '%s.pth' % scan_id))
    obj_labels = json.load(open(os.path.join(scan_dir, 'instance_id_to_name', '%s.json' % scan_id)))

    origin_pcds = []
    batch_pcds = []
    batch_labels = []
    inst_locs = []
    obj_ids = []
    for i, obj_label in enumerate(obj_labels):
        if (not keep_background) and obj_label in ['wall', 'floor', 'ceiling']:
            continue
        mask = instance_labels == i
        assert np.sum(mask) > 0, 'scan: %s, obj %d' % (scan_id, i)
        obj_pcd = pcds[mask]
        obj_color = colors[mask]
        origin_pcds.append(np.concatenate([obj_pcd, obj_color], 1))

        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))

        height_array = obj_pcd[:, 2:3] - obj_pcd[:, 2:3].min()

        # normalize
        obj_pcd = obj_pcd - obj_pcd.mean(0)
        max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, 1)))
        if max_dist < 1e-6:     # take care of tiny point-clouds, i.e., padding
            max_dist = 1
        obj_pcd = obj_pcd / max_dist
        obj_color = obj_color / 127.5 - 1

        # sample points
        pcd_idxs = np.random.choice(len(obj_pcd), size=2048, replace=(len(obj_pcd) < 2048))
        obj_pcd = obj_pcd[pcd_idxs]
        obj_color = obj_color[pcd_idxs]
        obj_height = height_array[pcd_idxs]

        # if use_scannet20:
        #     # Map the category name to id
        #     label_ids = labels_pd[labels_pd['raw_category'] == obj_label]['id']
        #     label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0

        #     # obj_label = list(VALID_CLASS_IDS_200).index(label_id)
        #     obj_label_200 = CLASS_LABELS_200[list(VALID_CLASS_IDS_200).index(label_id)]

        batch_pcds.append(np.concatenate([obj_pcd, obj_height, obj_color], 1))
        batch_labels.append(obj_label)
        obj_ids.append(i)

    batch_pcds = torch.from_numpy(np.stack(batch_pcds, 0))
    center = (pcds.max(0) + pcds.min(0)) / 2

    return batch_labels, obj_ids, inst_locs, center, batch_pcds


def load_pred_ins(scan_id, normalize=True, use_scannet200=True):
    root_dir = '/221019046/Data/Mask3d/scannet'
    if use_scannet200:
        root_dir = '/221019046/Data/Mask3d/scannet200'
    data = np.load(os.path.join(root_dir, scan_id + '.npz'), allow_pickle=True)
    batch_labels = data['ins_labels']

    batch_pcds = []
    inst_locs = []
    scene_pc = []

    for i, obj in enumerate(data['ins_pcds']):
        if obj.shape[0] == 0:
            obj = np.zeros((1, 6))
        obj_pcd = obj[:, :3]
        scene_pc.append(obj_pcd)
        obj_color = obj[:, 3:6]
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))

        height_array = obj_pcd[:, 2:3] - obj_pcd[:, 2:3].min()

        # normalize
        if normalize:
            obj_pcd = obj_pcd - obj_pcd.mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, 1)))
            if max_dist < 1e-6:     # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd = obj_pcd / max_dist
            obj_color = obj_color / 127.5 - 1

        # sample points
        pcd_idxs = np.random.choice(len(obj_pcd), size=2048, replace=(len(obj_pcd) < 2048))
        obj_pcd = obj_pcd[pcd_idxs]
        obj_color = obj_color[pcd_idxs]
        obj_height = height_array[pcd_idxs]

        batch_pcds.append(np.concatenate([
            obj_pcd,
            obj_height,
            obj_color,
        ], 1))

    batch_pcds = torch.from_numpy(np.stack(batch_pcds, 0))
    scene_pc = np.concatenate(scene_pc, 0)
    center = (scene_pc.max(0) + scene_pc.min(0)) / 2

    return batch_labels, inst_locs, center, batch_pcds