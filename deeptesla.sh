# deeptesla.sh extracts videos and steering values from deep-tesla/epochs 
#  and stores them in a new directory in root /deeptesla
# all the images are in the same directory and they are stored under
# the name "dataset_num"_image_"still_num"_"steer_value"
# make sure this file is placed in deep-tesla/epochs

echo *Make sure that this file is placed in deep-tesla/epochs!!
echo *Else ^C now!!
sleep 2 
echo *Creating deeptesla directory
rm -rf ../../deeptesla
mkdir ../../deeptesla
array=($(ls *.mkv | grep -o [0-9][0-9]))
echo "*Dataset will be stored here: "$(cd ../../deeptesla && pwd)
for DATASET_NUM in "${array[@]}"
do
    echo 
    echo "Setting up Dataset#""$DATASET_NUM"
    #sed -i '1d' epoch"$DATASET_NUM"_steering.csv
    STEER_ARRAY=($(awk -F "\"*,\"*" '{print $3}' epoch"$DATASET_NUM"_steering.csv)) 
    ffmpeg -i epoch"$DATASET_NUM"_front.mkv  ../../deeptesla/"$DATASET_NUM"_image%04d.png  
    var=1
    for STEER_VALUE in "${STEER_ARRAY[@]}"
    do
        name=$(seq -f "%04g" "$var" "$var")
        if [ -f ../../deeptesla/"$DATASET_NUM"_image"$name".png ]
        then
            mv ../../deeptesla/"$DATASET_NUM"_image"$name".png ../../deeptesla/"$DATASET_NUM"_image"$name"_"$STEER_VALUE".png
        else
            echo "$deeptesla/$DATASET_NUM_image$name.png not found!"
        fi
        var=$((var+1))
    done
    
done
echo 
echo *Process Completed...
echo *Self-Destruct deeptesla.sh in 5..
sleep 1 
echo *Self-Destruct deeptesla.sh in 4..
sleep 1 
echo *Self-Destruct deeptesla.sh in 3..
sleep 1 
echo *Self-Destruct deeptesla.sh in 2..
sleep 1 
echo *Self-Destruct deeptesla.sh in 1..
sleep 1 
rm -- "$0"
