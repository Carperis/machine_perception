import os
import shutil

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from diffusers.models import Transformer2DModel
from diffusers.models.unets import UNet2DConditionModel
from diffusers.models.transformers import SD3Transformer2DModel, FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from diffusers.models.attention import BasicTransformerBlock, JointTransformerBlock
from diffusers import FluxPipeline
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)

from modules import *

attn_maps = {}
feature_maps = {}

def cross_attn_init():
    # AttnProcessor.__call__ = attn_call
    # AttnProcessor2_0.__call__ = attn_call2_0
    # LoRAAttnProcessor.__call__ = lora_attn_call
    # LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    JointAttnProcessor2_0.__call__ = joint_attn_call2_0
    # FluxAttnProcessor2_0.__call__ = flux_attn_call2_0


def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):

            timestep = module.processor.timestep

            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_map = module.processor.attn_map.cpu() if detach else module.processor.attn_map
            attn_maps[timestep][name] = attn_map

            del module.processor.attn_map

        if hasattr(module.processor, "feature_map"):
            timestep = module.processor.timestep

            feature_maps[timestep] = feature_maps.get(timestep, dict())
            feature_map = module.processor.feature_map.cpu() if detach else module.processor.feature_map
            feature_maps[timestep][name] = feature_map

            del module.processor.feature_map

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        # if isinstance(module.processor, AttnProcessor):
        #     module.processor.store_attn_map = True
        # elif isinstance(module.processor, AttnProcessor2_0):
        #     module.processor.store_attn_map = True
        # elif isinstance(module.processor, LoRAAttnProcessor):
        #     module.processor.store_attn_map = True
        # elif isinstance(module.processor, LoRAAttnProcessor2_0):
        #     module.processor.store_attn_map = True
        elif isinstance(module.processor, JointAttnProcessor2_0):
            module.processor.store_attn_map = True
            module.processor.store_feature_map = True
        # elif isinstance(module.processor, FluxAttnProcessor2_0):
        #     module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_function(name))
    
    return model


# def replace_call_method_for_unet(model):
#     if model.__class__.__name__ == 'UNet2DConditionModel':
#         model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

#     for name, layer in model.named_children():

#         if layer.__class__.__name__ == 'Transformer2DModel':
#             layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)

#         elif layer.__class__.__name__ == 'BasicTransformerBlock':
#             layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)

#         replace_call_method_for_unet(layer)

#     return model


def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'JointTransformerBlock':
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)
        
        replace_call_method_for_sd3(layer)
    
    return model


# def replace_call_method_for_flux(model):
#     if model.__class__.__name__ == 'FluxTransformer2DModel':
#         model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

#     for name, layer in model.named_children():

#         if layer.__class__.__name__ == 'FluxTransformerBlock':
#             layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)

#         replace_call_method_for_flux(layer)

#     return model


def init_pipeline(pipeline):
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
        
    #     elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
    #         FluxPipeline.__call__ = FluxPipeline_call
    #         pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
    #         pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

    # else:
    #     if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
    #         pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
    #         pipeline.unet = replace_call_method_for_unet(pipeline.unet)

    return pipeline


def _save_batched_attention_maps(attn_map, total_tokens, upper_dir):
    to_pil = ToPILImage()
    for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
        batch_dir = os.path.join(upper_dir, f'batch-{batch}')
        if not os.path.exists(batch_dir):
            os.mkdir(batch_dir)

        startofword = True
        for i, (token, a) in enumerate(zip(tokens, attn[: len(tokens)])):
            if '</w>' in token:
                token = token.replace('</w>', '')
                if startofword:
                    token = '<' + token + '>'
                else:
                    token = '-' + token + '>'
                    startofword = True

            elif token != '<|startoftext|>' and token != '<|endoftext|>':
                if startofword:
                    token = '<' + token + '-'
                    startofword = False
                else:
                    token = '-' + token + '-'

            to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))

def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True):
    # unconditional: if True, only use the second half of the attention maps
    token_ids = tokenizer(prompts)['input_ids']
    total_tokens = []
    for token_id in token_ids:
        total_tokens.append(tokenizer.convert_ids_to_tokens(token_id))

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1) # [2, 24, 8, 8, 154] -> [2, 8, 8, 154]
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]  # [2, 8, 8, 154] -> [1, 8, 8, 154]
    total_attn_map = total_attn_map.permute(0, 3, 1, 2) # [1, 8, 8, 154] -> [1, 154, 8, 8]
    total_attn_map = torch.zeros_like(total_attn_map)  # [1, 154, 8, 8]
    total_attn_map_shape = total_attn_map.shape[-2:] # [8, 8]
    total_attn_map_number = 0

    for timestep, layers in attn_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        if not os.path.exists(timestep_dir):
            os.mkdir(timestep_dir)

        total_attn_map_step = torch.zeros_like(total_attn_map)
        total_attn_map_number_step = 0

        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            # if not os.path.exists(layer_dir):
            #     os.mkdir(layer_dir)

            attn_map = attn_map.sum(1).squeeze(1)  # [2, 24, 8, 8, 154] -> [2, 8, 8, 154]
            attn_map = attn_map.permute(0, 3, 1, 2)  # [2, 8, 8, 154] -> [2, 154, 8, 8]

            if unconditional:
                attn_map = attn_map.chunk(2)[1] # [2, 154, 8, 8] -> [1, 154, 8, 8]

            resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1

            total_attn_map_step += resized_attn_map
            total_attn_map_number_step += 1

            # _save_batched_attention_maps(attn_map, total_tokens, layer_dir)

        total_attn_map_step /= total_attn_map_number_step
        _save_batched_attention_maps(total_attn_map_step, total_tokens, timestep_dir)

    total_attn_map /= total_attn_map_number
    _save_batched_attention_maps(total_attn_map, total_tokens, base_dir)


def _save_batched_feature_maps(feature_map, total_tokens, upper_dir):
    for batch, (tokens, feature) in enumerate(zip(total_tokens, feature_map)):
        batch_dir = os.path.join(upper_dir, f'batch-{batch}')
        if not os.path.exists(batch_dir):
            os.mkdir(batch_dir)

        n = feature.shape[0]
        nx = n - 154
        x_feat = feature[:nx]
        c_feat = feature[nx:]
        torch.save(x_feat, os.path.join(batch_dir, 'x_feat.pt'))

        startofword = True
        for i, (token, c) in enumerate(zip(tokens, c_feat[: len(tokens)])):
            if '</w>' in token:
                token = token.replace('</w>', '')
                if startofword:
                    token = '<' + token + '>'
                else:
                    token = '-' + token + '>'
                    startofword = True

            elif token != '<|startoftext|>' and token != '<|endoftext|>':
                if startofword:
                    token = '<' + token + '-'
                    startofword = False
                else:
                    token = '-' + token + '-'
            torch.save(c, os.path.join(batch_dir, f'c_feat_{i}-{token}.pt'))

def save_feature_maps(feature_maps, tokenizer, prompts, base_dir='feature_maps', unconditional=True):
    token_ids = tokenizer(prompts)["input_ids"]
    total_tokens = []
    for token_id in token_ids:
        total_tokens.append(tokenizer.convert_ids_to_tokens(token_id))

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    total_feature_map = list(list(feature_maps.values())[0].values())[0]
    if unconditional:
        total_feature_map = total_feature_map.chunk(2)[1]
    total_feature_map = torch.zeros_like(total_feature_map)
    total_feature_map_number = 0

    for timestep, layers in feature_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        if not os.path.exists(timestep_dir):
            os.mkdir(timestep_dir)

        total_feature_map_step = torch.zeros_like(total_feature_map)
        total_feature_map_number_step = 0

        for layer, feature_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            # if not os.path.exists(layer_dir):
            #     os.mkdir(layer_dir)

            if unconditional:
                feature_map = feature_map.chunk(2)[1]

            total_feature_map += feature_map
            total_feature_map_number += 1

            total_feature_map_step += feature_map
            total_feature_map_number_step += 1

            # _save_batched_feature_maps(feature_map, total_tokens, layer_dir)

        total_feature_map_step /= total_feature_map_number_step
        _save_batched_feature_maps(total_feature_map_step, total_tokens, timestep_dir)

    total_feature_map /= total_feature_map_number
    _save_batched_feature_maps(total_feature_map, total_tokens, base_dir)
