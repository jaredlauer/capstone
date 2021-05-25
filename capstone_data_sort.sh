#!/bin/bash

# This script sorts the cancer images into separate folders based on their class
# labels, and creates an 80/20 train/test split
#
# Folder structure:
#
# +--capstone
#    +--train
#    |   +--malignant   (80% of class 1 images)
#    |   +--benign      (80% of class 0 images)
#    +--test
#        +--malignant   (20% of class 1 images)
#        +--benign      (20% of class 0 images)

# create data folders
train_malignant_path="/Users/jaredlauer/Documents/BrainStation/Capstone/capstone/train/malignant/"
train_benign_path="/Users/jaredlauer/Documents/BrainStation/Capstone/capstone/train/benign/"

test_malignant_path="/Users/jaredlauer/Documents/BrainStation/Capstone/capstone/test/malignant/"
test_benign_path="/Users/jaredlauer/Documents/BrainStation/Capstone/capstone/test/benign/"

csv_path="/Users/jaredlauer/Documents/BrainStation/Capstone/capstone/kaggle_data/train_labels.csv"

echo "Making directory $train_malignant_path"
mkdir -p "$train_malignant_path"

echo "Making directory $train_non_malignant_path"
mkdir -p "$train_benign_path"

echo "Making directory $test_malignant_path"
mkdir -p "$test_malignant_path"

echo "Making directory $test_non_malignant_path"
mkdir -p "$test_benign_path"

malignant_counter=1
benign_counter=1

# sorting loop
IFS=","
while read filename label
do
    echo "Filename is : $filename and label is : $label"
    image_path="/Users/jaredlauer/Documents/BrainStation/Capstone/capstone/kaggle_data/train/$filename.tif"

    # check if file exists
    if [ -f "$image_path" ]; then
        # file exists!
        if [ "$label" == "1" ]; then

            # Place every 5th image into the test folder
            if [ "$malignant_counter" -gt 4 ]; then
                # Reset counter
                ((malignant_counter=1))

                echo "    Copying $filename to malignant test folder"
                cp "$image_path" "$test_malignant_path"

            else
                # Increment counter
                ((malignant_counter++))

                echo "    Copying $filename to malignant train folder"
                cp "$image_path" "$train_malignant_path"
            fi
        else

            if [ "$benign_counter" -gt 4 ]; then
                # Reset counter
                ((benign_counter=1))

                echo "    Copying $filename to benign test folder"
                cp "$image_path" "$test_benign_path"

            else
                # Increment counter
                ((benign_counter++))

                echo "    Copying $filename to benign train folder"
                cp "$image_path" "$train_benign_path"
            fi
        fi
    else
        # file does not exist
        echo "    File $image_path does not exist."
    fi

done < $csv_path # .csv file to loop over

echo "----FINISHED COPYING!----"

# count files
train_malignant_count=$(ls $train_malignant_path | wc -l)
echo "Number of malignant images in the train folder: $train_malignant_count"

train_benign_count=$(ls $train_benign_path | wc -l)
echo "Number of benign images in the train folder: $train_benign_count"

test_malignant_count=$(ls $test_malignant_path | wc -l)
echo "Number of malignant images in the test folder: $test_malignant_count"

test_benign_count=$(ls $test_benign_path | wc -l)
echo "Number of benign images in the test folder: $test_benign_count"
