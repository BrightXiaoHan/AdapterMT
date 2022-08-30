# Copyright (c) Lan-Bridge, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask


@dataclass
class TranslationAdapterConfig(TranslationConfig):
    adapter_dims: int = field(
        default=None,  # use the same dim as the ffn size
        metadata={"help": "dimension size of the adapter layers"},
    )
    phrase: str = field(
        default="adapter",
        metadata={"help": "training phrase, 'full' for full training and 'adapter' for training adapter layer only."},
    )

@register_task("translation_adapter", dataclass=TranslationAdapterConfig)
class TranslationAdapterTask(TranslationTask):
    """
    Translation task with extra adapter layers to adapte different domain or languages.

    See `"Simple, Scalable Adaptation for Neural Machine Translation"
    (Bapna et al., 2019) <https://arxiv.org/abs/1909.08478>`_.
    """

    cfg: TranslationAdapterConfig

    def __init__(self, cfg: TranslationAdapterConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def build_model(self, cfg, from_checkpoint=False):
        model = super().build_model(cfg, from_checkpoint=from_checkpoint)
        # freeze all the layers except the adapter layers
        if cfg.phrase == "adapter":
            for p in model.parameters():
                p.requires_grad = False
            for layer in model.encoder.layers:
                layer.adapter_layer1.parameters().requires_grad = True
                layer.adpater_layer2.parameters().requires_grad = True

            for layer in model.decoder.layers:
                layer.adapter_layer1.parameters().requires_grad = True
                layer.adpater_layer2.parameters().requires_grad = True
        return model
