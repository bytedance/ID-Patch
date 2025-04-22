# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch
from pathlib import Path
import numpy as np
from diffusers import ControlNetModel, EulerDiscreteScheduler
from diffusers.loaders.unet import UNet2DConditionLoadersMixin

from .pipeline_idpatch_sd_xl import StableDiffusionXLIDPatchPipeline

class IDPatchInferencer:
    def __init__(self, base_model_path, idp_model_path, patch_size=64, torch_device='cuda:0', torch_dtype=torch.bfloat16):
        super().__init__()
        self.patch_size = patch_size
        self.torch_device = torch_device
        self.torch_dtype = torch_dtype
        idp_state_dict = torch.load(Path(idp_model_path) / 'id-patch.bin', map_location="cpu")
        loader = UNet2DConditionLoadersMixin()
        self.id_patch_projection = loader._convert_ip_adapter_image_proj_to_diffusers(idp_state_dict['patch_proj']).to(self.torch_device, dtype=self.torch_dtype).eval()
        self.id_prompt_projection = loader._convert_ip_adapter_image_proj_to_diffusers(idp_state_dict['prompt_proj']).to(self.torch_device, dtype=self.torch_dtype).eval()
        controlnet = ControlNetModel.from_pretrained(Path(idp_model_path) / 'ControlNetModel').to(self.torch_device, dtype=self.torch_dtype).eval()
        scheduler = EulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
        self.pipe = StableDiffusionXLIDPatchPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            scheduler=scheduler,
            torch_dtype=self.torch_dtype,
        ).to(self.torch_device)
        
    def get_text_embeds_from_strings(self, text_strings):
        pipe = self.pipe
        device = pipe.device
        tokenizer_1 = pipe.tokenizer
        tokenizer_2 = pipe.tokenizer_2
        text_encoder_1 = pipe.text_encoder
        text_encoder_2 = pipe.text_encoder_2
        
        text_embeds = []
        for tokenizer, text_encoder in [(tokenizer_1, text_encoder_1), (tokenizer_2, text_encoder_2)]:
            input_ids = tokenizer(
                text_strings,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            text_embeds.append(text_encoder(input_ids, output_hidden_states=True))
        pooled_embeds = text_embeds[1]['text_embeds']
        text_embeds = torch.concat([text_embeds[0]['hidden_states'][-2], text_embeds[1]['hidden_states'][-2]], dim=2)
        return text_embeds, pooled_embeds
    
    def generate(self, face_embeds, face_locations, control_image, prompt, negative_prompt="", guidance_scale=5.0, num_inference_steps=50, controlnet_conditioning_scale=0.8, id_injection_ratio=0.8, seed=-1):
        """
            face_embeds: n_faces x 512
            face_locations: n_faces x 2[xy]
            control_image: PIL image
        """
        
        face_locations = face_locations.to(self.torch_device, self.torch_dtype)
        control_image = torch.from_numpy(np.array(control_image)).to(self.torch_device, dtype=self.torch_dtype).permute(2,0,1)[None] / 255.0
        height, width = control_image.shape[2:4]
        
        text_embeds, pooled_embeds = self.get_text_embeds_from_strings([negative_prompt, prompt]) # text_embeds: 2 x 77 x 2048, pooled_embeds: 2 x 1280
        negative_pooled_embeds, pooled_embeds = pooled_embeds[:1], pooled_embeds[1:]
        negative_text_embeds, text_embeds = text_embeds[:1], text_embeds[1:]
        
        n_faces = len(face_embeds)
        negative_id_embeds = self.id_prompt_projection(torch.zeros(n_faces, 1, 512, device=self.torch_device, dtype=self.torch_dtype)) # (BxF) x 16 x 2048
        negative_id_embeds = negative_id_embeds.reshape(1, -1, negative_id_embeds.shape[2]) # B x (Fx16) x 2048
        negative_text_id_embeds = torch.concat([negative_text_embeds, negative_id_embeds], dim=1)
        
        face_embeds = face_embeds[None].to(self.torch_device, self.torch_dtype) # 1 x faces x 512
        id_embeds = self.id_prompt_projection(face_embeds.reshape(-1, 1, 512)) # (BxF) x 16 x 2048
        id_embeds = id_embeds.reshape(face_embeds.shape[0], -1, id_embeds.shape[2]) # B x (Fx16) x 2048
        text_id_embeds = torch.concat([text_embeds, id_embeds], dim=1) # B x (77+Fx16) x 2048
        
        patch_prompt_embeds = self.id_patch_projection(face_embeds.reshape(-1, 1, 512)) # (Bxn_faces) x 3 x (64*64)
        patch_prompt_embeds = patch_prompt_embeds.reshape(1, n_faces, 3, self.patch_size, self.patch_size)
        pad = self.patch_size // 2
        canvas = torch.zeros((1, 3, height + pad * 2, width + pad * 2), device=self.torch_device)
        
        xymin = torch.round(face_locations - self.patch_size // 2).int()
        xymax =torch.round(face_locations + self.patch_size // 2).int()
        for f in range(n_faces):
            xmin, ymin = xymin[f,0], xymin[f,1]
            xmax, ymax = xymax[f,0], xymax[f,1]
            if xmin+pad < 0 or xmax-pad >= width or ymin+pad < 0 or ymax-pad >= height:
                continue
            canvas[0,:,ymin+pad:ymax+pad,xmin+pad:xmax+pad] += patch_prompt_embeds[0,f]
        condition_image = control_image + canvas[:,:,pad:-pad,pad:-pad]
        
        if seed >= 0:
            generator = torch.Generator(self.torch_device).manual_seed(seed)
        else:
            generator = None
        output_image = self.pipe(
            prompt_embeds=text_id_embeds,
            pooled_prompt_embeds=pooled_embeds,
            negative_prompt_embeds=negative_text_id_embeds,
            negative_pooled_prompt_embeds=negative_pooled_embeds,
            image=condition_image,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            id_injection_ratio=id_injection_ratio,
            output_type='pil',
            generator=generator,
        ).images[0]
        return output_image