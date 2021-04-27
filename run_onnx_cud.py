import argparse
import numpy as np
import onnxruntime as ort

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    X = np.zeros((1, 2048, 16, 16)).astype(np.float32)
    # X = np.zeros((1, 3, 512, 512)).astype(np.float32)
    model = ort.InferenceSession(args.model)
    input_name = model.get_inputs()[0].name
    class_name = model.get_outputs()[0].name
    bbox_name = model.get_outputs()[1].name
    pred_onx = model.run([class_name, bbox_name], {input_name: X})