import glob
import os, json
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, Blip2Processor, Blip2ForConditionalGeneration

from data.scannet200_constants import CLASS_LABELS_200, CLASS_LABELS_20
from zsvg.registry import register_interpreter
from zsvg.program import parse_step


def load_pred_ins(scan_id, normalize=True, use_scannet200=False):
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


@register_interpreter
class LocInterpreter(nn.Module):
    step_name = 'LOC_3D_pred'

    def __init__(self, tokenizer_name='/221019046/Data/huggingface/clip-vit-base-patch16'):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=512, use_fast=True)
        self.clip = CLIPModel.from_pretrained(tokenizer_name).cuda()

        self.class_name_list = list(CLASS_LABELS_200)
        self.class_name_list.remove('wall')
        self.class_name_list.remove('floor')
        self.class_name_list.remove('ceiling')

        self.class_name_tokens = self.tokenizer([f'a {class_name} in a scene' for class_name in self.class_name_list],
                                                padding=True,
                                                return_tensors='pt')
        for name in self.class_name_tokens.data:
            self.class_name_tokens.data[name] = self.class_name_tokens.data[name].cuda()

        label_lang_infos = self.clip.get_text_features(**self.class_name_tokens)
        self.label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)
        output_var = parse_result['output_var']

        obj_name = eval(parse_result['args']['object'])
        scan_id = prog_step.state['scan_id']

        pred_labels, pred_locs, center, pred_pcds = load_pred_ins(scan_id, use_scannet200=True)
        boxes = self.predict(pred_labels, pred_locs, obj_name, pred_pcds, scan_id)

        prog_step.state[output_var] = boxes
        prog_step.state['CENTER'] = [{'obj_id': -1, 'obj_loc': center, 'obj_name': 'CENTER'}]
        return boxes

    def predict(self, pred_labels, inst_locs, obj_name, batch_pcds, scan_id):
        boxes = []

        new_class_list = list(set(pred_labels))
        new_class_name_tokens = self.tokenizer([f'a {class_name} in a scene' for class_name in new_class_list],
                                               padding=True,
                                               return_tensors='pt')

        for name in new_class_name_tokens.data:
            new_class_name_tokens.data[name] = new_class_name_tokens.data[name].cuda()
        label_lang_infos = self.clip.get_text_features(**new_class_name_tokens)
        label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)

        query_name_tokens = self.tokenizer([f'a {obj_name} in a scene'], padding=True, return_tensors='pt')
        for name in query_name_tokens.data:
            query_name_tokens.data[name] = query_name_tokens.data[name].cuda()

        query_lang_infos = self.clip.get_text_features(**query_name_tokens)
        query_lang_infos = query_lang_infos / query_lang_infos.norm(p=2, dim=-1, keepdim=True)

        text_cls = torch.matmul(query_lang_infos, label_lang_infos.t())
        text_cls = text_cls.argmax(dim=-1)[0]
        text_cls = new_class_list[text_cls]

        for i in range(len(pred_labels)):

            if pred_labels[i] == text_cls:
                boxes.append({'obj_id': i, 'obj_name': text_cls, 'obj_loc': inst_locs[i]})

        return boxes


@register_interpreter
class BLIPInterpreter(nn.Module):
    step_name = 'LOC_BLIP_pred'

    def __init__(self, use2d=True, tokenizer_name='/221019046/Data/huggingface/clip-vit-base-patch16'):
        super().__init__()

        self.use2d = use2d

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=512, use_fast=True)
        self.clip = CLIPModel.from_pretrained(tokenizer_name).cuda()

        model_name = "/221019046/Data/huggingface/blip2-flan-t5-xl"
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

        self.class_name_list = list(CLASS_LABELS_200)
        self.class_name_list.remove('wall')
        self.class_name_list.remove('floor')
        self.class_name_list.remove('ceiling')

        self.class_name_tokens = self.tokenizer([f'a {class_name} in a scene' for class_name in self.class_name_list],
                                                padding=True,
                                                return_tensors='pt')
        for name in self.class_name_tokens.data:
            self.class_name_tokens.data[name] = self.class_name_tokens.data[name].cuda()

        label_lang_infos = self.clip.get_text_features(**self.class_name_tokens)
        self.label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)

        self.image_path = '/LiZhen_team/dataset/scannet/'

    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)
        output_var = parse_result['output_var']

        obj_name = eval(parse_result['args']['object'])
        scan_id = prog_step.state['scan_id']

        pred_labels, pred_locs, center, pred_pcds = load_pred_ins(scan_id, use_scannet200=True)
        boxes = self.predict(pred_labels, pred_locs, obj_name, pred_pcds, scan_id)

        prog_step.state[output_var] = boxes
        prog_step.state['CENTER'] = [{'obj_id': -1, 'obj_loc': center, 'obj_name': 'CENTER'}]
        return boxes

    def predict(self, pred_labels, inst_locs, obj_name, batch_pcds, scan_id):
        boxes = []

        new_class_list = list(set(pred_labels))
        new_class_name_tokens = self.tokenizer([f'a {class_name} in a scene' for class_name in new_class_list],
                                               padding=True,
                                               return_tensors='pt')

        for name in new_class_name_tokens.data:
            new_class_name_tokens.data[name] = new_class_name_tokens.data[name].cuda()
        label_lang_infos = self.clip.get_text_features(**new_class_name_tokens)
        label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)

        query_name_tokens = self.tokenizer([f'a {obj_name} in a scene'], padding=True, return_tensors='pt')
        for name in query_name_tokens.data:
            query_name_tokens.data[name] = query_name_tokens.data[name].cuda()

        query_lang_infos = self.clip.get_text_features(**query_name_tokens)
        query_lang_infos = query_lang_infos / query_lang_infos.norm(p=2, dim=-1, keepdim=True)

        text_cls = torch.matmul(query_lang_infos, label_lang_infos.t())
        text_cls = text_cls.argmax(dim=-1)[0]
        text_cls = new_class_list[text_cls]

        candidate_boxes = []

        for i in range(len(pred_labels)):
            obj_id = i
            if pred_labels[i] == text_cls:
                img_file = glob.glob(
                    os.path.join(self.image_path, '2d_bbox_pred_200_crop_1x', scan_id, str(obj_id), 'img_*.jpg'))

                imgs = []
                for image in img_file:
                    image = Image.open(image)
                    imgs.append(image)

                if len(img_file) > 0 and self.use2d:
                    prompt = [f"Question: Is there a {obj_name}? Answer:"] * len(img_file)
                    inputs = self.processor(images=imgs, text=prompt, return_tensors="pt").to('cuda', torch.float16)

                    generated_ids = self.model.generate(**inputs)
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                    if "yes" in generated_text:
                        # ic(obj_id, pred_scores, pred_class)
                        boxes.append({'obj_id': obj_id, 'obj_name': text_cls, 'obj_loc': inst_locs[i]})

                else:
                    boxes.append({'obj_id': obj_id, 'obj_name': text_cls, 'obj_loc': inst_locs[i]})

                candidate_boxes.append({'obj_id': obj_id, 'obj_name': text_cls, 'obj_loc': inst_locs[i]})

        if len(boxes) == 0:
            boxes = candidate_boxes

        return boxes
