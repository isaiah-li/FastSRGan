import tensorflow as tf
from tensorflow import keras
import  tensorflow_model_optimization as tfmot

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
      layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}

# ########量化自定义的Keras层
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope

class ModifiedDenseQuantizeConfig(DefaultDenseQuantizeConfig):
    # Configure weights to quantize with 4-bit instead of 8-bits.
    def get_weights_and_quantizers(self, layer):
      return [(layer.kernel, LastValueQuantizer(num_bits=4, symmetric=True, narrow_range=False, per_axis=False))]

class ModifiedDenseQuantizeConfig_2(DefaultDenseQuantizeConfig):
    def get_activations_and_quantizers(self, layer):
      # Skip quantizing activations.
      return []

    def set_quantize_activations(self, layer, quantize_activations):
      # Empty since `get_activaations_and_quantizers` returns
      # an empty list.
      return

class CustomLayer(tf.keras.layers.Dense):
  pass

model = quantize_annotate_model(tf.keras.Sequential([
    #tf.keras.layers.Conv2D(filters=20,kernel_size=3),
   quantize_annotate_layer(CustomLayer(20, input_shape=(20,)), ModifiedDenseQuantizeConfig_2()),
    tf.keras.layers.ReLU(),
   tf.keras.layers.Flatten()
]))
model.summary()
# `quantize_apply` requires mentioning `DefaultDenseQuantizeConfig` with `quantize_scope`
# as well as the custom Keras layer.
with quantize_scope(
    #{'ModifiedDenseQuantizeConfig': ModifiedDenseQuantizeConfig,'CustomLayer': CustomLayer}):
    {'ModifiedDenseQuantizeConfig_2': ModifiedDenseQuantizeConfig_2,
                    'CustomLayer': CustomLayer}):
  # Use `quantize_apply` to actually make the model quantization aware.
  quant_aware_model = tfmot.quantization.keras.quantize_apply(model)

quant_aware_model.summary()




