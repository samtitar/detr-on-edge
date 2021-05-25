import time
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

    inputs = [e.name for e in model.get_inputs()]
    outputs = [e.name for e in model.get_outputs()]

    st = time.time()
    pred_onx = model.run(outputs, {inputs[0]: X})
    print(time.time() - st)