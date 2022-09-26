# export HOME=/home/ipol
exec_root=`pwd`

# touch stdout.txt

# pwd
# echo HELLO

# mkdir -p $bin/input/
# cp input_0.png $bin/input/

cd $bin
# ls
# ls ./input/

python src/inference/inference_image.py -w weights/FFraw.tar -i $exec_root/input_0.png
    # >> $bin/stdout.txt

# mv output.png $exec_root
echo Successful!
