import numpy as np
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers import BartConfig
from transformers.models.bart.modeling_bart  import BartDecoder

class TinyDalleModel(BartForConditionalGeneration):
  def reinit_model_for_images(self):
    self.lm_head=nn.Linear(in_features=1024, out_features=16384+1, bias=False)
    self.final_logits_bias=torch.rand(16384+1)
    
    self.decoder=BartDecoder(BartConfig())
    self.decoder.padding_idx=16385
    self.decoder.offset=0
    self.decoder.embed_scale=math.sqrt(1024)
    self.decoder.embed_positions=nn.Embedding(256,1024)
    self.decoder.embed_tokens=nn.Embedding(16384+1,1024)
    
    self.encoder.embed_tokens=nn.Embedding(50264,1024)