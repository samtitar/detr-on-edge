import argparse
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 1

    with open(onnx_file_path, 'rb') as f:
        parser.parse(f.read())

    last_layer = network.get_layer(network.num_layers - 1)
    network.mark_output(last_layer.get_output(0))

    return builder.build_cuda_engine(network)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    inputs = np.random.random((1, 3, 512, 512)).astype(np.float32)

    engine = build_engine(args.model)
    context = engine.create_execution_context()

    for binding in engine:
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
            device_input = cuda.mem_alloc(input_size)
        else:
            output_shape = engine.get_binding_shape(binding)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)