import os
from utils import os_utils, img_utils
import cv2
import numpy as np
from preprocessing.detecion_models.insightface_2018.src.align import align_custom
from tqdm import tqdm
import pandas as pd
import argparse


def squarify(M,val=0):
    (a, b, c)=M.shape
    if a==b:
        return M
    if a>b:
        diff = a-b
        left_diff = diff // 2
        right_diff = diff - left_diff
        padding=((0,0), (left_diff,right_diff), (0,0))
    else:
        diff = b-a
        top_diff = diff // 2
        btm_diff = diff - top_diff
        padding=((top_diff, btm_diff), (0,0), (0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_partition', type=int, default=1)
    parser.add_argument('--partition_idx', type=int, default=0)
    parser.add_argument('--preprocess_method', type=str, default='pad_0.2',
                                                         choices=("", 'pad_0.1', 'pad_0.2'))
    parser.add_argument('--failed_image_behavior', type=str, default='center_crop',
                                                             choices=('identity', 'center_crop', 'pad_high'))
    parser.add_argument('--image_root', type=str, default='./tinyface/tinyface/Testing_Set')
    parser.add_argument('--save_root', type=str, default='./aligned')

    args = parser.parse_args()
    save_dir = os.path.join(args.save_root, "aligned_{}_{}".format(args.preprocess_method, args.failed_image_behavior))

    image_paths = os_utils.get_all_files(args.image_root, extension_list=['.jpg', '.png', '.jpeg'])
    print('total images: {}'.format(len(image_paths)))

    aligner = align_custom.Aligner()
    all_img_index = range(len(image_paths))

    if args.num_partition > 1:
        dataset_split = np.array_split(image_paths, args.num_partition)
        image_paths = list(dataset_split[args.partition_idx])
        all_img_index_split = np.array_split(all_img_index, args.num_partition)
        all_img_index = list(all_img_index_split[args.partition_idx])

    assert len(all_img_index) == len(image_paths)
    # batchfiy
    failed_list = []
    batchsize = 32
    n_batches = -(-len(image_paths) // batchsize)
    print("{} / {}".format(args.num_partition, args.partition_idx), 
            'num images', len(image_paths), 'batchsize', batchsize, 'n_batches', n_batches)
    batched_image_paths = np.array_split(image_paths, n_batches, axis=0)
    batched_image_idx = np.array_split(all_img_index, n_batches, axis=0)

    for img_index_list, paths in tqdm(zip(batched_image_idx, batched_image_paths), total=len(batched_image_paths)):

        images = []
        for path in paths.tolist():
            # input is rgb image
            image = aligner.img_read(path)
            image = img_utils.convert_image_type(image, dtype=np.uint8)

            if image.ndim == 2:
                image = align_custom.to_rgb(image)
            if image.ndim > 3:
                image = image[:, :, 0:3]

            square_img = squarify(image)
            if square_img.shape[0] != 32:
                square_img = cv2.resize(square_img, (32,32))
            
            if 'pad' in args.preprocess_method:
                pad_ratio = float(args.preprocess_method.split('_')[-1])
                pad_side = int(pad_ratio * 32)
                padding=((pad_side, pad_side), (pad_side,pad_side), (0,0))
                square_img = np.pad(square_img, padding,mode='constant',constant_values=0)

            images.append(square_img)
        images = np.stack(images, axis=0)

        # aligned
        res, aligned_images = aligner.infer_align_batch(images, return_image=True,
                                                        failed_image_behavior=args.failed_image_behavior)
        failed_index = [i for i, r in zip(img_index_list, res) if r is None]
        failed_list.extend(failed_index)
        
        # save
        for path, img in zip(paths, aligned_images):
            save_path = path.replace(args.image_root, save_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)

    pd.DataFrame(pd.Series(failed_list)).to_csv(
        os.path.join(args.save_root,'failed_tinyface_index_{}_{}.csv'.format(args.num_partition, args.partition_idx)))