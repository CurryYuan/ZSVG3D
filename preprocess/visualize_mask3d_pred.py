import numpy as np
import os
import torch
from data.scannet200_constants import SCANNET_COLOR_MAP_200, CLASS_LABELS_200, VALID_CLASS_IDS_200
from tqdm import tqdm
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene


scan_id_file = "data/scannet/splits/scannetv2_val.txt"
scene_list = set([x.strip() for x in open(scan_id_file, 'r')])
scene_list = sorted(list(scene_list))

pc_path = 'data/processed/scannet/validation'
pred_path = 'eval_output/instance_evaluation_scannet200_val_query_150_topk_750_dbscan_0.95_0/decoder_-1'
scan_dir = '/LiZhen_team/dataset/scannet/scans'


for scene_id in tqdm(scene_list):
    if scene_id == 'scene0647_00':
        pc = np.load(os.path.join(pc_path, scene_id[-7:] + '.npy'))

        align_matrix = np.eye(4)
        with open(os.path.join(scan_dir, scene_id, '%s.txt' % (scene_id)), 'r') as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
                    break
        # Transform the points
        pts = np.ones((pc.shape[0], 4), dtype=np.float32)
        pts[:, 0:3] = pc[:, 0:3]
        coords = np.dot(pts, align_matrix.transpose())[:, :3]  # Nx4
        # Make sure no nans are introduced after conversion
        assert (np.sum(np.isnan(coords)) == 0)
        pc[:, 0:3] = coords

        ins_pcds = []
        pred_info = np.genfromtxt(os.path.join(pred_path, scene_id + '.txt'), dtype='str')
        for i in range(pred_info.shape[0]):
            path = pred_info[i, 0]
            pred_class = pred_info[i, 1]
            pred_score = pred_info[i, 2]

            if float(pred_score) < 0.01:
                continue

            pred_ins = np.loadtxt(os.path.join(pred_path, path))

            ins = pc[pred_ins.astype(bool), :6]
            ins[:, 3:] = SCANNET_COLOR_MAP_200[int(pred_class)]
            ins_pcds.append(ins)

        break

print(len(ins_pcds))
pc = np.concatenate(ins_pcds, axis=0)
print(pc.shape)

points = torch.tensor(pc[:, :3]).unsqueeze(0).cuda()
colors = torch.tensor(pc[:, 3:]).unsqueeze(0).cuda() / 255

point_cloud = Pointclouds(points=points, features=colors)

plot_scene({"Pointcloud": {"scene0011_00": point_cloud}}, pointcloud_max_points=100000)