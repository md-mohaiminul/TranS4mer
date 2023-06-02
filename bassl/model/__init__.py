# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from .crn.trn import TransformerCRN
from .shot_encoder.resnet import resnet50
from .simplical_transformer import SimplicalTransformer
# from .TimeSformer.timesformer.models.vit import TimeSformer
# from .timm.models.vision_transformer import _create_vision_transformer
from .timm.models import create_model
import torch

def get_shot_encoder(cfg):
    name = cfg.MODEL.shot_encoder.name
    shot_encoder_args = cfg.MODEL.shot_encoder[name]
    if name == "resnet":
        depth = shot_encoder_args["depth"]
        if depth == 50:
            shot_encoder = resnet50(
                pretrained=shot_encoder_args["use_imagenet_pretrained"],
                **shot_encoder_args["params"],
            )
        else:
            raise NotImplementedError
    elif name == 'videoswin':
        shot_encoder = TimeSformer(img_size=shot_encoder_args["img_size"], num_classes=shot_encoder_args["num_classes"],
                                   num_frames=shot_encoder_args["num_frames"], attention_type=shot_encoder_args["attention_type"])

    elif name == 'vit':
        #shot_encoder = timm.create_model('vit_small_patch32_224', pretrained=True, num_classes=0)   #vit_small_patch16_224   vit_base_patch32_224
        shot_encoder = create_model('vit_small_patch32_224', pretrained=True, num_classes=0,
                                    attention_type = shot_encoder_args["attention_type"])

    else:
        raise NotImplementedError

    return shot_encoder


def get_contextual_relation_network(cfg):
    crn = None

    if cfg.MODEL.contextual_relation_network.enabled:
        name = cfg.MODEL.contextual_relation_network.name
        crn_args = cfg.MODEL.contextual_relation_network.params[name]
        if name == "trn":
            sampling_name = cfg.LOSS.sampling_method.name
            crn_args["neighbor_size"] = (
                2 * cfg.LOSS.sampling_method.params[sampling_name]["neighbor_size"]
            )
            crn = TransformerCRN(crn_args)
        else:
            raise NotImplementedError
    return crn

def get_sim_trans(cfg):
    sim_trans = SimplicalTransformer(d_model = 768)
    return sim_trans

__all__ = ["get_shot_encoder", "get_contextual_relation_network", "get_sim_trans"]
