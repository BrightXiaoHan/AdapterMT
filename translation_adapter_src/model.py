import torch.nn as nn
from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model
from fairseq.models.transformer.transformer_decoder import \
    TransformerDecoderBase
from fairseq.models.transformer.transformer_encoder import \
    TransformerEncoderBase
from fairseq.models.transformer.transformer_legacy import TransformerModel
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_layer import (TransformerDecoderLayerBase,
                                               TransformerEncoderLayerBase)


class TransformerEncoderLayerWithAdapter(TransformerEncoderLayerBase):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.adapter_dims = cfg.adapter_dims if cfg.adapter_dims else self.embed_dim
        self.adapter_layer1 = nn.Linear(self.embed_dim, self.adapter_dims)
        self.adapter_layer2 = nn.Linear(self.embed_dim, self.adapter_dims)

    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        residual = x
        x = self.activation_fn(self.adapter_layer1(x))
        x = self.activation_dropout_module(x)
        x = self.adapter_layer1(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        return x


class TransformerDecoderLayerWithAdapter(TransformerDecoderLayerBase):
    
        def __init__(self, cfg, *args, **kwargs):
            super().__init__(cfg, *args, **kwargs)
            self.adapter_dims = cfg.adapter_dims if cfg.adapter_dims else self.embed_dim
            self.adapter_layer1 = nn.Linear(self.embed_dim, self.adapter_dims)
            selk.adapter_layer2 = nn.Linear(self.embed_dim, self.adapter_dims)
    
        def forward(self, x, *args, **kwargs):
            x, attn, self_attn_state = super().forward(x, *args, **kwargs)
            residual = x
            x = self.activation_fn(self.adapter_layer1(x))
            x = self.activation_dropout_module(x)
            x = self.adapter_layer1(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            return x, attn, self_attn_state


class TransformerEncoderWithAdapter(TransformerEncoderBase):
    def build_encoder_layer(self, cfg):
        layer = TransformerEncoderLayerWithAdapter(
            cfg, return_fc=self.return_fc
        )
        # below is copied from fairseq.models.transformer.transformer_encoder.TransformerEncoder.build_encoder_layer
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class TransformerDecoderWithAdapter(TransformerDecoderBase):
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = TransformerDecoderLayerWithAdapter(cfg, no_encoder_attn)
        # below is copied from fairseq.models.transformer.transformer_decoder.TransformerDecoder.build_decoder_layer
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


@register_model("transformer_adapter")
class TransformerWithAdapterModel(TransformerModel):
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )
