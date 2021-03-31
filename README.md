# K210_mnist
It's a Handwritten digital recognition demo based on K210.Model modify by LeNet .Using MNIST dataset for training.

Firmware is used for MaixPy development board and MaixPy IDE,in my project ,I'm using the MAIX DOCK. 

The toolbox include nncase and some necessary files.(in this demo some python script must use tensorflow1 such as tensorflow 1.15)

The model folder include the demo model(trained by tensorflow2) and the compressed model which is adapted by K210 processor.

The scr folder include "lenet_k210.py" --> model build and train script. 
                       "h52pb.py" --> h5 to pb file
                       "mnist1.py" --> MaixPy IDE script
                       "dataset2jpg.py" --> convert the MNIST dataset to images and save them(provide images during compression)
                       
                       
1 :install the firmware (using kflash_gui) on your development board. Also you need to install the MaixPy IDE(more information look at https://wiki.sipeed.com)

2 :install tensorflow environment  build and train the model(save as .h5) then use script convert .h5 file to .pb file

3 :using code to generate images from dataset 

4 :using toolbox to convert .pb file to tflite 
   (./pb2tflite.sh )
   
5 :compress tflite model to kmodel
   (./ncc/ncc compile workspace/mnist_lenet_K210_model.tflite workspace/test1.kmodel -i tflite -o kmodel -t k210 --dataset images --dump-ir --max-allocator-solve-secs 60 --calibrate-method l2 --inference-type uint8 -v)
   (more information about nncase you can see https://github.com/kendryte/nncase )
   
6 :you can use 'infer' command to check you kmodel if you want ('infer' is the nnace command just like 'compile')
  (./ncc/ncc infer workspace/test1.kmodel workspace/test1.bin --dataset images)
  
7 :using kflash_gui to download kmodel into the development board in a particular location (like 0x300000 in this demo)

8 :run MaixPy IDE script to debug you code and download into flash or save the script at the SD card.

if you want to see the compute graph, you can run this command (python ./gen_pb_graph.py workspace/mnist_lenet_K210_model.pb). The compute graph can show model input layer name and the final output layer name.The compute graph is base on TensorBoard.

For more details you can see https://blog.csdn.net/weixin_44874976/article/details/104487069?spm=1001.2014.3001.5501
