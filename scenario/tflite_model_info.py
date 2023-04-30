import tensorflow as tf
import os

def tflite_model_info(args):
    file_path = os.path.join(args.save_path, args.tflite_file_name)
    interpreter = tf.lite.Interpreter(file_path)

    input_details = interpreter.get_input_details()
    print(input_details)

    output_details = interpreter.get_output_details()
    print(output_details)
