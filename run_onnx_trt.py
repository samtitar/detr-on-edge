import time
import torch
import argparse
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def build_engine(onnx_file_path):
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print('Parsing onnx file')
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    
    print('Setting configuration')
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20

    print('Setting optimizaiton profile')
    profile = builder.create_optimization_profile()
    # profile.set_shape('image', (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
    profile.set_shape('vector', (1, 2048, 7, 7), (1, 2048, 7, 7), (1, 2048, 7, 7))
    config.add_optimization_profile(profile)

    print('Building engine')
    engine = builder.build_engine(network, config)
    print('Engine built')
    return engine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    engine = build_engine(args.model)
    context = engine.create_execution_context()

    # Backbone
    # inputs = [torch.ones((1, 3, 224, 224), device="cuda:0")]
    # outputs = [torch.zeros((1, 2048, 7, 7), device="cuda:0")]

    # Interpreter
    inputs = [torch.zeros((1, 2048, 7, 7), device="cuda:0")]
    outputs = [torch.zeros((1, 100, 92), device="cuda:0"),
               torch.zeros((1, 100, 4), device="cuda:0")]

    bindings = [_input.data_ptr() for _input in inputs] + [_output.data_ptr() for _output in outputs]

    s_time = time.time()
    context.execute_v2(bindings)
    print(time.time() - s_time)
    print(outputs[0].shape, outputs[1].shape)
