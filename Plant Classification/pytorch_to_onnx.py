import argparse

def Pytorch_to_ONNX(pytorch_model, onnx_model_dir):
    pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pytorch_model_dir',
        default="results",
        help='Directory of the pytorch(.pt) model.')
    parser.add_argument(
        '--onnx_model_dir',
        default="results",
        help='Directory of the ONNX(.onnx) model.')


    args, _ = parser.parse_known_args()

    print(f"-------- Start of convertion to ONNX ---------")
    
    Pytorch_to_ONNX(args.pytorch_model_dir, args.onnx_model_dir)

    print(f"-------- End of conversion to ONNX --------")
