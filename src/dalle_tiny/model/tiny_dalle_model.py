import numpy as np
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers import BartConfig
from transformers.models.bart.modeling_bart  import BartDecoder
import math

class TinyDalleModel(BartForConditionalGeneration):
  def reinit_model_for_images(self):
    self.lm_head=nn.Linear(in_features=1024, out_features=16384+1, bias=False)
    self.final_logits_bias=torch.rand(16384+1)    
    t=BartDecoder(BartConfig())
    self.get_decoder().layers=t.layers
    self.get_decoder().layernorm_embedding=t.layernorm_embedding
    self.get_decoder().padding_idx=16385
    self.get_decoder().offset=0
    self.get_decoder().embed_scale=math.sqrt(1024)
    # self.get_decoder().embed_positions=nn.Embedding(256,1024)
    self.get_decoder().embed_tokens=nn.Embedding(16384+1,1024)    
    # self.get_encoder().embed_tokens=nn.Embedding(50264,1024)
    self.config.decoder_start_token_id=16384
    del t