import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import config

from model.transformer_layers import Decoder, Encoder
from model.attn_class import AttnEmo


class EmotionInputDecoder(nn.Module):
    def __init__(self, emo_input):
        super(EmotionInputDecoder, self).__init__()
        self.emo_input = emo_input
        if self.emo_input == "self_att":#default
            self.enc = Encoder(2 * config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)
        elif self.emo_input == "cross_att":
            self.dec = Decoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)
        elif self.emo_input == "att":
            self.dec = AttnEmo(config.emb_dim)
        else:
            raise ValueError("Invalid attention mode.")
            
    def forward(self, emotion, encoder_outputs, mask_src):
        repeat_vals = [-1] + [encoder_outputs.shape[1] // emotion.shape[1]] + [-1]
        if self.emo_input == "self_att": #default
            hidden_state_with_emo = torch.cat([encoder_outputs, emotion.expand(repeat_vals)], dim=2)
            aux = self.enc(hidden_state_with_emo, mask_src)
            return aux
        elif self.emo_input == "cross_att":
            return self.dec(emotion, encoder_outputs, (None, mask_src))[0]
        elif self.emo_input == "att":
            return self.dec(encoder_outputs, emotion.expand(repeat_vals), mask_src)




