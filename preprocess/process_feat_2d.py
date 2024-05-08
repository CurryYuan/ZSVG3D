import pickle
import os, sys
import cv2
import math
import plyfile
import numpy as np
from transformers import CLIPModel, AutoProcessor
from tqdm import tqdm
import time
from functools import partial
import glob
from PIL import Image
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_pc


def read_aggregation(filename):
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1     # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(root_path, scene_id):
    agg_file = os.path.join(root_path, 'scans', scene_id, f'{scene_id}.aggregation.json')
    seg_file = os.path.join(root_path, 'scans', scene_id, f'{scene_id}_vh_clean_2.0.010000.segs.json')

    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)

    object_ids = np.zeros(shape=(num_verts), dtype=np.uint32)     # 0: unannotated
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            object_ids[verts] = object_id
    return object_ids



class PointCloudToImageMapper(object):

    def __init__(self, visibility_threshold=0.25, cut_bound=0, intrinsics=None):

        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, camera_to_world, coords, depth, image_dim, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None:     # global intrinsics
            intrinsic = self.intrinsics

        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int)     # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < image_dim[0]-self.cut_bound) \
                    * (pi[1] < image_dim[1]-self.cut_bound)

        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                                    self.vis_thres * depth_cur

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0     # make sure the depth is in front
            inside_mask = front_mask * inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1

        return mapping.T


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def computeLinking(camera_to_world, coords, depth, intricsic, image_dim):
    """
    :param camera_to_world: 4 x 4
    :param coords: N x 3 format
    :param depth: H x W format
    :return: linking, N x 3 format, (H,W,mask)
    """
    link = np.zeros((3, coords.shape[0]), dtype=np.int)
    coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
    assert coordsNew.shape[0] == 4, "[!] Shape error"

    world_to_camera = np.linalg.inv(camera_to_world)
    p = np.matmul(world_to_camera, coordsNew)
    p[0] = (p[0] * intricsic[0][0]) / p[2] + intricsic[0][2]
    p[1] = (p[1] * intricsic[1][1]) / p[2] + intricsic[1][2]
    pi = np.round(p).astype(int)
    inside_mask = (pi[0] >= 0) * (pi[1] >= 0) \
                  * (pi[0] < image_dim[0]) * (pi[1] < image_dim[1])

    occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= 0.1
    inside_mask[inside_mask] = occlusion_mask

    link[0][inside_mask] = pi[1][inside_mask]
    link[1][inside_mask] = pi[0][inside_mask]
    link[2][inside_mask] = 1

    return link.T



def load_img(f):
    img = cv2.imread(f)
    image_dim = (img.shape[1], img.shape[0])

    # img = cv2.resize(img, (image_dim[0], image_dim[1]), interpolation=cv2.INTER_NEAREST)

    depth = cv2.imread(f.replace('color', 'depth').replace('jpg', 'png'), cv2.IMREAD_UNCHANGED) / 1000.0     # convert to meter
    depth = cv2.resize(depth, image_dim, interpolation=cv2.INTER_NEAREST)

    posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
    pose = np.loadtxt(posePath)

    return img, depth, pose, image_dim


def process_scan(scene_id, root_path, image_root, point2img_mapper, processor, clip):

    # print(f'Processing {scene_id} ...')
    image_dir = os.path.join(image_root, scene_id, 'color')
    image_list = os.listdir(image_dir)
    # print('Number of images: {}.'.format(len(image_list)))

    # load intrinsic parameter
    intrinsics = np.loadtxt(os.path.join(image_root, scene_id, 'intrinsic_color.txt'))

    os.makedirs(os.path.join(root_path, '2d_bbox', scene_id), exist_ok=True)

    # load point cloud
    coord = plyfile.PlyData().read(os.path.join(root_path, 'scans', scene_id, f'{scene_id}_vh_clean_2.ply'))
    v = np.array([list(x) for x in coord.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])     # (N, 3)

    # load instance label
    ins_labels = export(root_path, scene_id)     # (N, )

    batch_labels, obj_ids, inst_locs, center, batch_pcds = load_pc(scene_id)

    imgs = {}
    areas = {}

    for image_name in image_list:
        f = os.path.join(image_dir, image_name)
        img, depth, pose, image_dim = load_img(f)

        for obj_id in obj_ids:

            pc = coords[ins_labels == (obj_id + 1)]

            link = np.ones([pc.shape[0], 4], dtype=int)
            link[:, 1:4] = point2img_mapper.compute_mapping(pose, pc, depth, image_dim, intrinsics)

            link = link[:, 1:]
            valid_map = link[link[:, -1] != 0]

            empty = np.zeros((image_dim[1], image_dim[0], 3))
            empty[valid_map[:, 0], valid_map[:, 1]] = np.array([255, 255, 255])     #colors[link[:, -1] != 0]

            map = img.copy()
            map[valid_map[:, 0], valid_map[:, 1]] = np.array([255, 255, 255])

            # indices = np.nonzero(empty)
            indices = np.nonzero(empty != 0)

            # filter out points less than 10
            if len(indices[0]) < 20:
                continue

            crop_h = indices[0].max() - indices[0].min()
            crop_w = indices[1].max() - indices[1].min()

            if crop_h < 20 or crop_w < 20:
                continue

            crop_img = img[max(indices[0].min() - crop_h // 4, 0):indices[0].max() + crop_h // 4,
                           max(indices[1].min() - crop_w // 4, 0):indices[1].max() + crop_w // 4]

            map_img = map[max(indices[0].min() - crop_h // 4, 0):indices[0].max() + crop_h // 4,
                          max(indices[1].min() - crop_w // 4, 0):indices[1].max() + crop_w // 4]

            if str(obj_id) not in imgs:
                imgs[str(obj_id)] = []
                areas[str(obj_id)] = []
            imgs[str(obj_id)].append((crop_img, map_img, image_name))
            areas[str(obj_id)].append(crop_h * crop_w)

    img_feats = {}
    for obj_id, area in areas.items():
        # select top-k
        k = 5
        idx = np.argsort(area)[-k:]
        img_list = [imgs[obj_id][i] for i in idx]

        inputs = processor(images=[img[0] for img in img_list], return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        img_feat = clip.get_image_features(**inputs)
        img_feats[str(obj_id)] = img_feat.detach().cpu()

    return img_feats


def process_scan_pred(scene_id, root_path, image_root, point2img_mapper, processor, clip):

    # print(f'Processing {scene_id} ...')
    image_dir = os.path.join(image_root, scene_id, 'color')
    image_list = os.listdir(image_dir)
    # print('Number of images: {}.'.format(len(image_list)))

    # load point cloud
    root_dir = '/221019046/Data/Mask3d/scannet200_unalign'
    data = np.load(os.path.join(root_dir, scene_id + '.npz'), allow_pickle=True)
    batch_labels = data['ins_labels']

    imgs = {}
    areas = {}

    for image_name in image_list:
        f = os.path.join(image_dir, image_name)
        img, depth, pose, image_dim = load_img(f)

        for obj_id, pc in enumerate(data['ins_pcds']):
            pc = pc[:, :3]

            link = np.ones([pc.shape[0], 4], dtype=int)
            link[:, 1:4] = point2img_mapper.compute_mapping(pose, pc, depth, image_dim)

            link = link[:, 1:]
            valid_map = link[link[:, -1] != 0]

            empty = np.zeros((image_dim[1], image_dim[0], 3))
            empty[valid_map[:, 0], valid_map[:, 1]] = np.array([255, 255, 255])     #colors[link[:, -1] != 0]

            map = img.copy()
            map[valid_map[:, 0], valid_map[:, 1]] = np.array([255, 255, 255])

            # indices = np.nonzero(empty)
            indices = np.nonzero(empty != 0)

            # filter out points less than 20
            if len(indices[0]) < 20:
                continue

            crop_h = indices[0].max() - indices[0].min()
            crop_w = indices[1].max() - indices[1].min()

            if crop_h < 50 or crop_w < 50:
                continue

            crop_img = img[max(indices[0].min() - crop_h // 4, 0):indices[0].max() + crop_h // 4,
                           max(indices[1].min() - crop_w // 4, 0):indices[1].max() + crop_w // 4]

            # crop_img = img[max(indices[0].min(), 0):indices[0].max(), max(indices[1].min(), 0):indices[1].max()]

            map_img = map[max(indices[0].min() - crop_h // 4, 0):indices[0].max() + crop_h // 4,
                          max(indices[1].min() - crop_w // 4, 0):indices[1].max() + crop_w // 4]

            if str(obj_id) not in imgs:
                imgs[str(obj_id)] = []
                areas[str(obj_id)] = []
            imgs[str(obj_id)].append((crop_img, map_img, image_name))
            areas[str(obj_id)].append(crop_h * crop_w)

    img_feats = {}
    for obj_id, area in areas.items():
        # select top-k
        k = 5
        idx = np.argsort(area)[-k:]
        img_list = [imgs[obj_id][i] for i in idx]

        imgs = processor(images=imgs, return_tensors="pt", padding=True)
        imgs['pixel_values'] = imgs['pixel_values'].cuda()
        img_feat = clip.get_image_features(**imgs)
        img_feats[f'{scene_id}_{obj_id}'] = img_feat


def main():
    root_path = '/LiZhen_team/dataset/scannet/'
    image_root = os.path.join(root_path, 'scannet_2d')

    scan_id_file = "data/scannet/splits/scannetv2_val.txt"
    scene_list = list(set([x.strip() for x in open(scan_id_file, 'r')]))
    scene_list = list(filter(lambda x: x.endswith('00'), scene_list))
    # scene_list = ['scene0011_00']

    tokenizer_name = '/221019046/Data/huggingface/clip-vit-base-patch16'
    clip = CLIPModel.from_pretrained(tokenizer_name).cuda()
    # clip.eval()
    processor = AutoProcessor.from_pretrained(tokenizer_name)

    visibility_threshold = 0.25     # threshold for the visibility check
    cut_num_pixel_boundary = 0     # do not use the features on the image boundary

    # calculate image pixel-3D points correspondances
    point2img_mapper = PointCloudToImageMapper(intrinsics=None,
                                               visibility_threshold=visibility_threshold,
                                               cut_bound=cut_num_pixel_boundary)

    process_func = partial(process_scan,
                           root_path=root_path,
                           image_root=image_root,
                           point2img_mapper=point2img_mapper,
                           processor=processor,
                           clip=clip)

    # process_func = partial(process_scan_pred,
    #                        root_path=root_path,
    #                        image_root=image_root,
    #                        point2img_mapper=point2img_mapper,
    #                        predictor=None)

    feats_2d = {}
    for scene_id in tqdm(scene_list):
        feats = process_func(scene_id)
        feats_2d[scene_id] = feats

    pickle.dump(feats_2d, open('data/scannet/feats_2d.pkl', 'wb'))


if __name__ == '__main__':

    main()
