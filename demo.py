# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from mtcnn import MTCNN
from insightface.utils import face_align
import facexlib
import torch
from modules.inferencer import IDPatchInferencer
from rtmlib import Body
from utils.draw_condition import draw_openpose_from_mmpose


def mtcnn_to_kps(mtcnn_results):
    kps = np.array([mtcnn_results[0]['keypoints']['left_eye'], mtcnn_results[0]['keypoints']['right_eye'], mtcnn_results[0]['keypoints']['nose'], mtcnn_results[0]['keypoints']['mouth_left'], mtcnn_results[0]['keypoints']['mouth_right']])
    return kps

def extract_face_emb(arcface_encoder, cropped_face):
    face_image = torch.from_numpy(cropped_face).unsqueeze(0).permute(0,3,1,2) / 255.
    face_image = 2 * face_image - 1
    face_image = face_image.to('cuda:0').contiguous()
    face_emb = arcface_encoder(face_image)[0]
    return face_emb
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_dir', default='data/subjects')
    parser.add_argument('--output_path', default='result.png')
    parser.add_argument('--reference_image_path', default='data/reference_image.png')
    parser.add_argument('--prompt', default="a young couple in front of their burning home still managing to find a moment of joy amidst disaster. cheerfully raise glasses filled with a bright blue drink")
    parser.add_argument('--negative_prompt', default="nude, worst quality, low quality, normal quality, nsfw, abstract, glitch, deformed, mutated, ugly, disfigured, text, watermark, bad hands, error, jpeg artifacts, blurry, missing fingers")
    parser.add_argument('--guidance_scale', type=float, default=5.5)
    parser.add_argument('--controlnet_conditioning_scale', type=float, default=0.8)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--id_injection_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--idp_model_path', default='models/ID-Patch')
    parser.add_argument('--base_model_path', default='RunDiffusion/Juggernaut-X-v10')
    args = parser.parse_args()
    
    tf.config.set_visible_devices([], 'GPU')
    mtcnn_inferencer = MTCNN() # MTCNN might be slow, could be replaced by other face detectors, as long as it provides 5 keypoints
    # load subjects
    face_embs = []
    arcface_encoder = facexlib.recognition.init_recognition_model('arcface', device='cuda:0')
    for subject_path in sorted(list(Path(args.subject_dir).iterdir())):
        image_subject = cv2.imread(str(subject_path))
        mtcnn_subject = mtcnn_inferencer.detect_faces(image_subject[:,:,::-1])
        kps_subject = mtcnn_to_kps(mtcnn_subject)
        cropped_face_subject = face_align.norm_crop(image_subject, landmark=kps_subject, image_size=112)
        face_embs.append(extract_face_emb(arcface_encoder, cropped_face_subject))
    face_embs = torch.stack(face_embs)
    
    # load style
    image_reference = cv2.imread(args.reference_image_path)
    mtcnn_reference = mtcnn_inferencer.detect_faces(image_reference[:,:,::-1])

    # estimate pose
    keypoints, scores = Body(to_openpose=False, mode='balanced', backend='onnxruntime', device='cpu')(image_reference)
    face_locations = keypoints[:,0]
    face_locations = torch.from_numpy(np.array(sorted(face_locations, key=lambda x: x[0])))
    
    idpatch_inferencer = IDPatchInferencer(base_model_path=args.base_model_path, idp_model_path=args.idp_model_path)
    control_image = Image.fromarray(draw_openpose_from_mmpose(image_reference * 0, keypoints, scores))

    result = idpatch_inferencer.generate(
        face_embs,
        face_locations,
        control_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        id_injection_ratio=args.id_injection_ratio,
        seed=args.seed
    )
    result.save(args.output_path)