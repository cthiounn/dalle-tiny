import numpy as np
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

class TinyDalleModel(BartForConditionalGeneration):
  def reinit_model_for_images(self):
    self.lm_head=nn.Linear(in_features=1024, out_features=16384+1, bias=False)
    self.final_logits_bias=torch.rand(16384+1)
    self.get_decoder().embed_tokens=nn.Embedding(16384+1,1024)
    self.get_encoder().embed_tokens=nn.Embedding(50264,1024)