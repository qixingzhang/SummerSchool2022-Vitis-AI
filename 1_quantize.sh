#!/bin/bash

# delete previous results
rm -rf ./quantization_output

# Quantize
echo "#####################################"
echo "QUANTIZE begin"
echo "#####################################"

python vitis_ai_tf2_quantize.py \
	--model float_model.h5 \
	--name quantized_model
