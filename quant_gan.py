from argparse import ArgumentParser
from tensorflow import keras
import numpy as np
import cv2
import os
import shutil
import tensorflow as tf

import tensorflow_model_optimization as tfmot

from main import train
from dataloader import DataLoader
from model import FastSRGAN

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory where images are kept.')
parser.add_argument('--output_dir', type=str, help='Directory where to output high res images.')
parser.add_argument('--model_name', type=str, help='model',default='models/generator.h5')
parser.add_argument('--train_dir', type=str, help='model',default='/mnt/SuperResolution/data/DIV2K_train_HR')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
parser.add_argument('--hr_size', default=384, type=int, help='Low resolution input size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')



def change_model(model, new_input_shape=(None, 40, 40, 3),custom_objects=None):
    # replace input shape of first layer
    
    config = model.layers[0].get_config()
    config['batch_input_shape']=new_input_shape
    model._layers[0]=model.layers[0].from_config(config)

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json(),custom_objects=custom_objects)

    # copy weights from old model to new one
    for layer in new_model._layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model

def solve_lambda(x):
    import tensorflow as tf
    result = tf.nn.depth_to_space(x,2)
    return result

def main():
    args = parser.parse_args()

    # Get all image paths
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]

    gen_modelpath = 'models/' + args.model_name
    dis_modelpath = 'models/' + args.model_name
    if not os.path.exists (gen_modelpath):
        assert ('please check the model dir')
    else:
        gen_modelpath = 'models/' + args.model_name + '/generator_' + args.model_name + '_800.h5'
        dis_modelpath = 'models/' + args.model_name + '/discriminator_' + args.model_name + '_800.h5'
        if not os.path.exists (gen_modelpath):
            assert ('please check the generator model path')
        if not os.path.exists (dis_modelpath):
            assert ('please check the discriminator model path')
    # Change model input shape to accept all size inputs
    #model = keras.models.load_model(gen_modelpath,custom_objects={'solve_lambda':solve_lambda})
    gen_model = keras.models.load_model(gen_modelpath)
    dis_model = keras.models.load_model(dis_modelpath)
    
    ### when the model trained by GPU,
    gen_model = change_model(gen_model, new_input_shape=[None, 96, 96, 3])
    dis_model = change_model(dis_model, new_input_shape=[None, 384, 384, 3])

    ##############################################################################
    # ############################## quantize the model ##########################    
    ##############################################################################
    quantize_model = tfmot.quantization.keras.quantize_model

    # q_aware stands for for quantization aware.
    gen_q_aware_model = quantize_model(gen_model)
    dis_q_aware_model = quantize_model(dis_model)

    # `quantize_model` requires a recompile.
    gen_q_aware_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    dis_q_aware_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    gen_q_aware_model.summary() 
    dis_q_aware_model.summary() 


    # ## save generator and discriminator model
    gen_q_aware_model.save('models/' + args.model_name + '/generator_' + args.model_name + '_quant.h5')
    dis_q_aware_model.save('models/' + args.model_name + '/discriminator_' + args.model_name + '_quant.h5')

    print('quantize end')
    print('Will FineTune ... ')

    # Create the tensorflow dataset.
    traindata = DataLoader(args.train_dir, args.hr_size).dataset(args.batch_size)

    # Initialize the GAN object.
    quant_model = FastSRGAN(args)
    quant_model.generator = gen_q_aware_model
    quant_model.discriminator = dis_q_aware_model

    # Define the directory for saving pretrainig loss tensorboard summary.
    pretrain_summary_writer = tf.summary.create_file_writer('logs/pretrain_{}'.format(args.model_name))

    # Run pre-training.
    # pretrain_generator(gan, ds, pretrain_summary_writer)

    # Define the directory for saving the SRGAN training tensorbaord summary.
    train_summary_writer = tf.summary.create_file_writer('logs/train_{}'.format(args.model_name))

    
    # Run training.
    for epochIndex in range(args.epochs):
        print('This epoch is {}, starting'.format(epochIndex))
        train(quant_model, traindata, args.save_iter, train_summary_writer,args.model_name,'fine_q_')
    print('FineTune end')





    # ############## test ####################
    
    inputs = keras.Input((None, None, 3))
    output = gen_q_aware_model(inputs)
    model = keras.models.Model(inputs, output)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Start inference')
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

        # Get super resolution image
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        # Rescale values in range 0-255
        sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

        # Convert back to BGR for opencv
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    
        
        # Save the results:
        cv2.imwrite(os.path.join(args.output_dir, os.path.basename(image_path)), sr)

        imgIndex += 1
    

if __name__ == '__main__':
    main()
