
set -e


# /workdir/bin
exec_root=`pwd`

# touch stdout.txt

# pwd
# echo HELLO

# mkdir -p $bin
# cp input_0.png $bin/input/

cd $bin

# ls
# ls ./input/

# pwd
# ls

# python src/inference/inference_image.py -w weights/FFraw.tar -i $exec_root/input_0.png
    # >> $bin/stdout.txt
    
python src/inference/inference_image.py \
    -w weights/FFraw.tar \
    -e weights/adv-efficientnet-b4-44fb3a87.pth \
    -r weights/retinaface_resnet50_2020-07-20.pth \
    -i $exec_root/input_0.png

# mv output.png $exec_root
echo Successful!
