import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        parser.parse(model.read())
    
    builder.max_workspace_size = 1 << 30
    builder.max_batch_size = 1

    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
        
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    return engine, context

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    engine, context = build_engine(args.model)
