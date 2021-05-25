import argparse
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def build_onnx_trt(model_path):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 1

    with open(model_path, 'rb') as f:
        parser.parse(f.read())

    last_layer = network.get_layer(network.num_layers - 1)
    network.mark_output(last_layer.get_output(0))

    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()

    def execute_trt(data):
        for binding in engine:
            if engine.binding_is_input(binding):
                input_shape = engine.get_binding_shape(binding)
                input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
                device_input = cuda.mem_alloc(input_size)
            else:
                output_shape = engine.get_binding_shape(binding)
                host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
                device_output = cuda.mem_alloc(host_output.nbytes)
    
    return execute_trt

def build_onnx_cud(model_path):
    import onnxruntime as ort

    model = ort.InferenceSession(model_path)

    inputs = [e.name for e in model.get_inputs()]
    outputs = [e.name for e in model.get_outputs()]

    def execute_cud(data):
        return model.run(outputs, {inputs[0]: data})

    return execute_cud

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, required=True)
    # parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--onnx-loc', type=str, required=True)
    parser.add_argument('--onnx-rem', type=str, required=True)
    parser.add_argument('--runtime-loc', type=str, required=True)
    parser.add_argument('--runtime-rem', type=str, required=True)
    args = parser.parse_args()

    if rank == 0:
        if args.runtime_loc == 'cud':
            model = build_onnx_cud(args.onnx_loc)
        if args.runtime_loc == 'trt':
            model = build_onnx_trt(args.onnx_loc)

        data = model(np.zeros((1, 3, 224, 224)).astype(np.float32))
        req = comm.isend(data, dest=1, tag=11)
        req.wait()
    elif rank == 1:
        if args.runtime_rem == 'cud':
            model = build_onnx_cud(args.onnx_rem)
        if args.runtime_rem == 'trt':
            model = build_onnx_trt(args.onnx_rem)

        req = comm.irecv(source=0, tag=11)
        data = req.wait()
        result = model(data)
        print(result)
