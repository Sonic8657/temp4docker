import ktc
import numpy as np
import os
import onnx
from PIL import Image
onnx_path = '/data1/temp4docker/yolov3-tiny-OP.onnx'
m = onnx.load(onnx_path)
# npu (only) performance simulation
km = ktc.ModelConfig(35000, "001a", "520", onnx_model=m)
eval_result = km.evaluate()
print("\nNpu performance evaluation result:\n" + str(eval_result))

from os import walk

img_list = []
for (dirpath, dirnames, filenames) in walk("/data1/temp4docker/voc_data50"):
    for f in filenames:
        fullpath = os.path.join(dirpath, f)
        
        image = Image.open(fullpath)
        image = image.convert("RGB")
        image = Image.fromarray(np.array(image)[...,::-1])
        img_data = np.array(image.resize((640, 640), Image.BILINEAR)) / 256 - 0.5
        print(fullpath)
        img_list.append(img_data)
        
# fixed-point analysis
bie_model_path = km.analysis({"input": img_list})
print("\nFixed-point analysis done. Saved bie model to '" + str(bie_model_path) + "'")

# compile
nef_model_path = ktc.compile([km])
print("\nCompile done. Saved Nef file to '" + str(nef_model_path) + "'")