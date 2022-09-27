# python src/inference/inference_video.py -w weights/FFraw.tar -i id0_id1_0000.mp4

# python src/inference/inference_image.py -w weights/FFraw.tar -i test.png


python src/inference/inference_image.py \
    -w weights/FFraw.tar \
    -e weights/adv-efficientnet-b4-44fb3a87.pth \
    -r weights/retinaface_resnet50_2020-07-20.pth \
    -i test.png