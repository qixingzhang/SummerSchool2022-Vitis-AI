import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow_model_optimization.quantization.keras import vitis_quantize
import sys
import os
import argparse

def main():
	parser = argparse.ArgumentParser(description='Vitis-AI Tensorflow2.x Quantize.') 
	parser.add_argument('-m', '--model', type=str, dest='model', help='h5 model', required=True)
	parser.add_argument('-n', '--name', type=str, dest='name', help='model name', default='quantized_model')
	parser.add_argument('-o', '--output', type=str, dest='output', help='output path', default='./quantization_output')
	args = parser.parse_args()

	if not os.path.exists(args.output):
		os.mkdir(args.output)

	model = args.model
 
	float_model = tf.keras.models.load_model(model)
	quantizer = vitis_quantize.VitisQuantizer(float_model)
	(train_img, train_label), (test_img, test_label) = mnist.load_data()
	test_img = test_img.reshape(-1, 28, 28, 1) / 255
	quantized_model = quantizer.quantize_model(calib_dataset=test_img)
	
	quantized_model.save(os.path.join(args.output, args.name+'.h5'))
 
	print('quantized model was saved to', args.output)
	print('#####################################')
	print('QUANTIZE COMPLETED')
	print('#####################################')

if __name__ == '__main__':
	main()
