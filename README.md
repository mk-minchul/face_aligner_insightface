

# Face Alignment for InsightFace
This code aligns images to the template as defined in insightface.

This code relies on tensorflow v1.

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
                      --failed_image_behavior center_crop \
                      --image_root ./tinyface/Testing_Set \
                      --save_root ./aligned
```