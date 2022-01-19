

# Face Alignment for InsightFace

## Installation


## Example Dataset
You can download and unzip tinyFace dataset from https://qmul-tinyface.github.io
```
# unzip in the root directory for below script
unzip tinyface.zip
```

## Alignment (example)
``` 
python align_mtcnn.py --num_partition 1 \
                      --partition_idx 0 \
                      --preprocess_method pad_0.2 \
                      --failed_image_behavior pad_high \
                      --image_root /data/data/faces/tinyface/Testing_Set \
                      --save_root ./aligned
```