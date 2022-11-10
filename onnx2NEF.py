import torch
import torch.onnx
import ktc
from PIL import Image
import numpy as np
import cv2

def preprocess(img_file_path):
    image = cv2.imread(img_file_path)
    # resize image to match model input size (this case: width 1024  height 512)
    image = cv2.resize(image , (256, 256,3), interpolation=cv2.INTER_LINEAR)
    # convert to numpy array
    np_data = np.array(image, dtype='float32')
    # data normalization (for OpenMMLab Kneron Edition, "pixel/256 - 0.5" )
    np_data = np_data/256.
    np_data = np_data - 0.5

    return np_data

km = ktc.ModelConfig(32769, "0001", "520", onnx_path="/workspace/temp4docker/resnet_50_Opt.onnx")
print('build km')
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
