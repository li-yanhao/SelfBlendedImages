# FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04
FROM registry.ipol.im/ipol:v1-py3.8-pytorch-gpu

# RUN apt-get -y update && \
    # apt-get -y install python3 python3-pip

# RUN apt-get update && apt-get upgrade -y && \
#     apt-get install -y libgl1-mesa-dev && \
#     apt-get install -y cmake && \
#     apt-get -y install python3-pip && \
#     apt-get -y update && apt-get install -y libopencv-dev \
#     apt-get install -y git
# RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
#     pip3 install dlib imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch&&\
#     pip3 install -U retinaface_pytorch &&\
#     apt-get clean

RUN pip install dlib imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch
RUN pip install -U retinaface-pytorch
RUN pip install gdown

# this is the path that $bin of IPOL server is connected to
ENV bin /workdir/bin
RUN mkdir -p $bin
workdir $bin


COPY . .

RUN mkdir -p weights

RUN gdown 12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV -O weights/FFraw.tar 

RUN wget https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth
RUN mv adv-efficientnet-b4-44fb3a87.pth weights/

RUN wget https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip
RUN unzip retinaface_resnet50_2020-07-20-f168fae3c.zip -d weights/ && rm retinaface_resnet50_2020-07-20-f168fae3c.zip


# RUN pwd
# RUN ls
# download model weights
# RUN bash test_img.sh

WORKDIR $bin

RUN groupadd -g 1000 ipol && useradd -m -u 1000 -g 1000 ipol -d $bin && chmod -R 777 $bin
USER ipol