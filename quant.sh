# 1
# python quant_gan.py --model_name '4K_initmodel'

# RuntimeError: Layer p_re_lu:<class 'tensorflow.python.keras.layers.advanced_activations.PReLU'> is not supported. 
# You can quantize this layer by passing a `tfmot.quantization.keras.QuantizeConfig` instance to the `quantize_annotate_layer` API.

# 2
# python quant_gan.py --model_name 'edsr_deconv_model'

# RuntimeError: Layer lambda:<class 'tensorflow.python.keras.layers.core.Lambda'> is not supported. 
# You can quantize this layer by passing a `tfmot.quantization.keras.QuantizeConfig` instance to the `quantize_annotate_laye` API.


python quant_gan.py --model_name 'fastsr_model' --image_dir '/mnt/SuperResolution/data/test_Internet' --output_dir 'result/fastsr_model_quant-4k' --train_dir '/mnt/SuperResolution/data/DIV2K_train_HR' --hr_size 384 --lr 1e-4 --save_iter 200 --epochs 3 --batch_size 8



