#!/bin/bash

# delete previous results
rm -rf ./compile_output

NET_NAME=MNIST_classifier

# Can be found in /opt/vitis_ai/compiler/arch/DPUCZDX8G
# ARCH="ultra96.json"
ARCH="kv260.json"

# Compile
echo "#####################################"
echo "COMPILE WITH DNNC begin"
echo "#####################################"
vai_c_tensorflow2 \
       --model ./quantization_output/quantized_model.h5 \
       --arch ${ARCH} \
       --output_dir compile_output \
       --net_name ${NET_NAME}

echo "#####################################"
echo "COMPILATION COMPLETED"
echo "#####################################"
