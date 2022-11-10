import torch
import torch.onnx
import ktc
from PIL import Image
import numpy as np


def preprocess(input_file):
    image = Image.open(input_file)
    image = image.convert("RGB")
    img_data = np.array(image.resize((256, 256), Image.BILINEAR)) / 255
    #img_data = np.transpose(img_data, (1, 0, 2))
    return img_data


km = ktc.ModelConfig(32769, "0001", "520", onnx_path="/workspace/temp4docker/resnet_50_Opt.onnx")
#eval_result = km.evaluate()

input_data = [preprocess("/workspace/temp4docker/image/0_webcam_1.jpg")]
inf_results = ktc.kneron_inference(input_data, onnx_file="/workspace/temp4docker/resnet_50_Opt.onnx", input_names=["data_out"])


# Preprocess images and create the input mapping
input_images = [
    preprocess("/workspace/temp4docker/image/0_webcam_1.jpg"),
    preprocess("/workspace/temp4docker/image/0_webcam_2.jpg"),
    preprocess("/workspace/temp4docker/image/0_webcam_3.jpg"),
    preprocess("/workspace/temp4docker/image/0_webcam_4.jpg"),
]
input_mapping = {"data_out": input_images}

# Quantization
bie_path = km.analysis(input_mapping, output_bie = None, threads = 4)

compile_result = ktc.compile([km])