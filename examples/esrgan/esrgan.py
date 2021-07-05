import tensorflow as tf
# import tensorflow_hub as hub
# import matplotlib.pyplot as plt
import cv2
import numpy as np
from argparse import ArgumentParser
import cv2
import shutil
from tensorflow import keras
import tensorflow_model_optimization as tfmot

parser = ArgumentParser()
parser.add_argument('--image', type=str, default='lr-1.jpg')
parser.add_argument('--initmodel', type=str, default = '../../models/fastsr_model/generator_fastsr_model_quant.h5')
parser.add_argument('--tflite', type=str, default = 'FASTRGAN_quant.tflite')


print(tf.__version__)


args = parser.parse_args()

# model = hub.load("https://hub.tensorflow.google.cn/captain-pool/esrgan-tf2/1")
# concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
# concrete_func.inputs[0].set_shape([1, 50, 50, 3])
# converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()

# Save the TF Lite model.
# with tf.io.gfile.GFile(args.tflite, 'wb') as f:
#   f.write(tflite_model)

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(args.initmodel)
model.layers[0].input.set_shape([None,96,96,3])
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(args.tflite, 'wb') as f:
    f.write(tflite_model)

####################################
esrgan_model_path = args.tflite

test_img_path = args.image
# lr = tf.io.read_file(test_img_path)
# lr = tf.image.decode_jpeg(lr)
# lr = tf.expand_dims(lr, axis=0)
# lr = tf.cast(lr, tf.float32)

lr = cv2.imread(test_img_path)
lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
lr = cv2.resize(lr,(96,96))
# Rescale to 0-1.
lr = lr / 255.0
lr =np.expand_dims(lr,axis = 0)
lr = lr.astype(np.float32)


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=esrgan_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run the model
interpreter.set_tensor(input_details[0]['index'], lr)
interpreter.invoke()

# Extract the output and postprocess it
# output_data = interpreter.get_tensor(output_details[0]['index'])
# sr = tf.squeeze(output_data, axis=0)
# sr = tf.clip_by_value(sr, 0, 255)
# sr = tf.round(sr)
# sr = tf.cast(sr, tf.uint8)

# lr = tf.cast(tf.squeeze(lr, axis=0), tf.uint8)

# sr = sr.numpy()

sr = interpreter.get_tensor(output_details[0]['index'])
sr = np.squeeze(sr)

    # Rescale values in range 0-255
sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

# Convert back to BGR for opencv

sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
srname = args.image.split('.')[0] + '_sr_quant.jpg'
cv2.imwrite(srname,sr)


# bicubic = tf.image.resize(lr, [200, 200], tf.image.ResizeMethod.BICUBIC)
# bicubic = tf.cast(bicubic, tf.uint8)
# bicubic = bicubic.numpy()
# bicubic = cv2.cvtColor(bicubic, cv2.COLOR_RGB2BGR)
# bicubicname = args.image.split('.')[0] + '_bicubic.jpg'
# cv2.imwrite(bicubicname,bicubic)



