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


with open('data/scannet/feats_3d.pkl', 'rb') as f:
    feats = pickle.load(f)

with open('data/scannet/feats_2d.pkl', 'rb') as f:
    feats_2d = pickle.load(f)


def load_pc(scan_id):
    obj_ids = feats[scan_id]['obj_ids']
    inst_locs = feats[scan_id]['inst_locs']
    center = feats[scan_id]['center']
    obj_embeds = feats[scan_id]['obj_embeds']

    return obj_ids, inst_locs, center, obj_embeds

@register_interpreter
class LocInterpreter(nn.Module):
    step_name = 'LOC'

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

        self.root_path = '/LiZhen_team/dataset/scannet/'


    def execute(self, prog_step, inspect=False):
        parse_result = parse_step(prog_step.prog_str)

        step_name = parse_result['step_name']
        assert (step_name == self.step_name)

        output_var = parse_result['output_var']

        obj_name = eval(parse_result['args']['object'])
        scan_id = prog_step.state['scan_id']

        obj_ids, inst_locs, center, obj_embeds = load_pc(scan_id)

        boxes = self.predict(obj_ids, inst_locs, obj_name, obj_embeds, scan_id)
        prog_step.state[output_var] = boxes
        prog_step.state['CENTER'] = [{'obj_id': -1, 'obj_loc': center, 'obj_name': 'CENTER'}]
        return boxes

    def predict(self, obj_ids, inst_locs, obj_name, obj_embeds, scan_id):
        boxes = []
        logit_scale = self.clip.logit_scale.exp()

        # cosine similarity as logits
        class_logits_3d = torch.matmul(self.label_lang_infos, obj_embeds.t())     # * logit_scale
        obj_cls = class_logits_3d.argmax(dim=0)
        pred_class_list = [self.class_name_list[idx] for idx in obj_cls]

        new_class_list = list(set(pred_class_list))
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

        for i in range(len(obj_ids)):

            if pred_class_list[i] == text_cls:
                obj_id = obj_ids[i]

                boxes.append({'obj_id': obj_ids[i], 'obj_name': text_cls, 'obj_loc': inst_locs[i]})

        return boxes


