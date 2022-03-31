import numpy as np
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers import BartConfig
from transformers.models.bart.modeling_bart  import BartDecoder
import math

class TinyDalleLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 0
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)
    
class TinyDalleModel(BartForConditionalGeneration):
  def reinit_model_for_images(self):
    self.lm_head=nn.Linear(in_features=1024, out_features=16384+1, bias=False)
    self.final_logits_bias=torch.rand(16384+1)    
    # t=BartDecoder(BartConfig())
    # self.get_decoder().layers=t.layers
    # self.get_decoder().layernorm_embedding=t.layernorm_embedding
    # self.get_decoder().padding_idx=16385
    # self.get_decoder().offset=0
    # self.get_decoder().embed_scale=math.sqrt(1024)
    # self.get_decoder().embed_positions=TinyDalleLearnedPositionalEmbedding(256,1024)
    # self.get_decoder().embed_tokens=nn.Embedding(16384+1,1024)
    # self.get_encoder().embed_tokens=nn.Embedding(50264,1024)
    # self.get_encoder().embed_positions=TinyDalleLearnedPositionalEmbedding(256,1024)
    # self.get_encoder().padding_idx=16385
    # self.get_encoder().offset=0
    # self.config.decoder_start_token_id=16384
    # del t