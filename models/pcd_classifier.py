import torch
import torch.nn as nn

from models.pointnext.pointnext import PointNEXT


class PcdClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_encoder = PointNEXT()
        self.pc_projection = nn.Parameter(torch.empty(256, 512))
    
    def forward(self, batch_pcds):
        obj_embeds = self.obj_encoder(batch_pcds[..., :4])
        obj_embeds = obj_embeds @ self.pc_projection

        return obj_embeds