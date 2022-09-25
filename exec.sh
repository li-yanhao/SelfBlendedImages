#!/bin/sh

# docker run -it --gpus all --shm-size 64G \
#     -v /path/to/this/repository:/app/ \
#     sbi bash


docker run -it --shm-size 8G \
    -v /Users/yli/phd/deepfake/SelfBlendedImages:/app/ \
    sbi bash