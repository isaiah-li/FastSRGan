import tensorflow as tf

import tensorflow_model_optimization as tfmot
import os
from argparse import ArgumentParser
import cv2
import shutil
import numpy as np

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')
#parser.add_argument('--model_name', type=str, help='model',default='models/generator.h5')

# #################### Keras convert to tflite ########################
keras_model_file = 'models/fastsr_model/fine_q_generator_fastsr_model_200.h5'
tflite_model_file = 'models/fastsr_model/finetune_quantized_fastsr.tflite'
# check the tflite_model_file exist or not
if not os.path.exists(tflite_model_file):
    with tfmot.quantization.keras.quantize_scope():
        quant_aware_model = tf.keras.models.load_model(keras_model_file)

    quant_aware_model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_tflite_model = converter.convert()


    with open(tflite_model_file, 'wb') as f:
        f.write(quantized_tflite_model)


# # tflite inference
interpreter = tf.lite.Interpreter(model_path = tflite_model_file)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()  
output_details = interpreter.get_output_details()



args = parser.parse_args()

args.image_dir = '/mnt/SuperResolution/data/test_Internet'
args.output_dir = 'result/fastsr_model-tflite'

# Get all image paths
image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]


if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print('Start tfllite inference')
imgIndex = 0
    # Loop over all images
for image_path in image_paths:
    print('this is the pic {}, and name is {}'.format(imgIndex,image_path))
    # Read image
    low_res = cv2.imread(image_path, 1)

    # Convert to RGB (opencv uses BGR as default)
    low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

    # Rescale to 0-1.
    low_res = low_res / 255.0

    low_res =np.expand_dims(low_res,axis = 0)
    low_res = low_res.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'],low_res)
    
    # #执行预测
    interpreter.invoke()
    
    sr = interpreter.get_tensor(output_details[0]['index'])
    sr = np.squeeze(sr)

    # Rescale values in range 0-255
    sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

    # Convert back to BGR for opencv
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        
    # Save the results:
    cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_path)), sr)

    imgIndex += 1

print('tflite model inference end')