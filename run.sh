
set -e


# /workdir/bin
exec_root=`pwd`

# touch stdout.txt

# pwd
# echo HELLO

# mkdir -p $bin
# cp input_0.png $bin/input/

#### IPOL ####
# cd $bin
##############

file=$1
echo $file

if [[ $file == *.png ]]; then
    python src/inference/inference_image.py \
    -w weights/FFraw.tar \
    -e weights/adv-efficientnet-b4-44fb3a87.pth \
    -r weights/retinaface_resnet50_2020-07-20.pth \
    -i $file
fi

if [[ $file == *.mp4 ]]; then
    python src/inference/inference_video.py \
    -w weights/FFraw.tar \
    -e weights/adv-efficientnet-b4-44fb3a87.pth \
    -r weights/retinaface_resnet50_2020-07-20.pth \
    -i $file
fi








# mv output.png $exec_root
echo Successful!
