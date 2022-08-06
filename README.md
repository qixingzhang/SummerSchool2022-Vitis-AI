# Vitis AI Lab: MNIST Classifier

## Install Vitis AI
1. Install docker engine
    * Official site: [https://docs.docker.com/engine/install/]()
    * For Ubuntu: [https://docs.docker.com/engine/install/ubuntu/]()
1. Pull Vitis AI 1.4 image
    ```
    sudo docker pull xilinx/vitis-ai:1.4.916
    ```
1. Clone from GitHub
    ```shell
    $ git clone https://github.com/Xilinx/Vitis-AI.git
    ```
    ```shell
    $ cd Vitis-AI
    ```
    ```shell
    $ git checkout v1.4
    ```
1. Clone this repository to `Vitis-AI` folder
    ```shell
    $ git clone
    ```
1. Launch Vitis AI
    ```shell
    sudo ./docker_run.sh xilinx/vitis-ai:1.4.916
    ```

## Vitis AI Workflow
1. Activate TensorFlow 2.x environment
    ```shell
    $ conda activate vitis-ai-tensorflow2
    ```
1. Train the model if needed
    > A pre-trained keras model (.h5 format) is provided, you can also train it by youself
    ```shell
    $ python train.py
    ```
1. Quantize the model
    ```shell
    $ chmod +x *.sh
    ```
    ```shell
    $ ./1_quantize.sh
    ```

    This script invokes the python script `vitis_ai_tf2_quantize.py`. It uses the python API of Vitis AI Quantizer:
    * `quantizer = vitis_quantize.VitisQuantizer(model)`
        * model: keras floating point model
    * `quantized_model = quantizer.quantize_model(calib_dataset=dataset)`
        * calib_dataset: dataset for calibration, 100 ~ 1000 training or testing images is enough.
1. Compile the model
    ```shell
    $ ./2_compile.sh
    ```
    This scripts uses the `vai_c_tensorflow2` commmand to compile the quantized model. It has four required parameters:
    * `--model` quantized model with h5 format
    * `--arch` indicate DPU arch, it can be found in `/opt/vitis_ai/compiler/arch/DPUCZDX8G` for different Zynq boards.
    * `--output_dir` compile output path
    * `--net_name` your model name

    The output `.xmodel` file is saved in `compile_output` folder.

## Deploy on edge board using DPU-PYNQ
1. Boot the board with PYNQ v2.7 image
    * Image download link: [http://www.pynq.io/board.html]()
1. Install python package
    ```shell
    $ sudo pip3 install pynq-dpu --no-build-isolation
    ```
1. Get notebooks
    ```shell
    $ cd $PYNQ_JUPYTER_NOTEBOOKS
    ```
    ```shell
    $ pynq get-notebooks pynq-dpu –p .
    ```
1. Run the `dpu_mnist_classifier` notebook in `pynq-dpu` folder. You can upload your own `.xmodel` file to this folder and replace the existing `dpu_mnist_classifier.xmodel`