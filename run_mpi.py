#!venv/bin/python3

import time
import argparse
import numpy as np
from mpi4py import MPI

BUFF_SIZE = 1024000

def build_onnx_trt(model_path, mode=0):
    import torch
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print('Parsing onnx file')
    with open(model_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    
    print('Setting configuration')
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20

    print('Setting optimizaiton profile')
    profile = builder.create_optimization_profile()

    if mode == 0:
        profile.set_shape('image', (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
    else:
        profile.set_shape('vector', (1, 2048, 7, 7), (1, 2048, 7, 7), (1, 2048, 7, 7))
    config.add_optimization_profile(profile)

    print('Building engine')
    engine = builder.build_engine(network, config)
    context = engine.create_execution_context()
    print('Engine built')

    def execute_trt(data):
        if mode == 0:
            inputs = [torch.tensort(data, device="cuda:0")]
            outputs = [torch.zeros((1, 2048, 7, 7), device="cuda:0")]
        else:
            inputs = [torch.tensor(data, device="cuda:0")]
            outputs = [torch.zeros((1, 100, 92), device="cuda:0"),
                       torch.zeros((1, 100, 4), device="cuda:0")]
        bindings = [i.data_ptr() for i in inputs] + [o.data_ptr() for o in outputs]
        context.execute_v2(bindings)

        return outputs
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
    parser.add_argument('--mode', type=int, required=True)
    parser.add_argument('--path-r0', type=str, required=True)
    parser.add_argument('--path-r1', type=str, required=True)
    parser.add_argument('--runtime-r0', type=str, required=True)
    parser.add_argument('--runtime-r1', type=str, required=True)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if args.runtime_loc == 'cud':
            model = build_onnx_cud(args.path_r0)
        if args.runtime_loc == 'trt':
            model = build_onnx_trt(args.path_r0)
        
        # Mode=0: run backbone on r0
        if args.mode == 0:
            # Get features from input
            data = np.zeros((1, 3, 224, 224)).astype(np.float32)

            stime = time.time()
            data = model(data)
            print(f'Backbone time: {time.time() - stime}')

            # Send features to remote
            req = comm.isend(data, dest=1, tag=11)
            req.wait()

            # Receive results from remote
            req = comm.irecv(BUFF_SIZE, source=1, tag=11)
            data = req.wait()
        elif args.mode == 1:
            # Send data to remote
            data = np.zeros((1, 3, 224, 224)).astype(np.float32)
            req = comm.isend(data, dest=1, tag=11)
            req.wait()

            # Receive features from remote
            req = comm.irecv(BUFF_SIZE, source=1, tag=11)
            data = req.wait()

            # Get results from features
            stime = time.time()
            data = model(data[0])
            print(f'Transformer time: {time.time() - stime}')
    elif rank == 1:
        if args.runtime_rem == 'cud':
            model = build_onnx_cud(args.onnx_rem)
        if args.runtime_rem == 'trt':
            model = build_onnx_trt(args.onnx_rem)

        # Receive data or features from remote
        req = comm.irecv(BUFF_SIZE, source=0, tag=11)
        data = req.wait()

        # Get features or results from data or features
        if args.mode == 0:
            stime = time.time()
            data = model(data[0])
            print(f'Backbone time: {time.time() - stime}')
        elif args.mode == 1:
            stime = time.time()
            data = model(data)
            print(f'Transformer time: {time.time() - stime}')

        # Send features or results to remote
        req =  comm.isend(data, dest=0, tag=11)
        req.wait()