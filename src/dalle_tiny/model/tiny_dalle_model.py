import numpy as np
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

class TinyDalleModel(BartForConditionalGeneration):
  def reinit_lm_head_for_images(self):
    self.lm_head=nn.Linear(in_features=1024, out_features=16384, bias=False)