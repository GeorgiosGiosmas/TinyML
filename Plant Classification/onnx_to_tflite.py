import argparse

def ONNX_to_TfLite(onnx_model, tflite_model_dir):
    pass

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--onnx_model_dir',
        default="results",
        help='Directory of the ONNX(.onnx) model.')
    parser.add_argument(
        '--tflite_model_dir',
        default="results",
        help='Directory of the TfLite(.tflite) model.')


    args, _ = parser.parse_known_args()

    print(f"-------- Start of convertion to TfLite ---------")
    
    ONNX_to_TfLite(args.onnx_model_dir, args.tflite_model_dir)

    print(f"-------- End of conversion to TfLite --------")
