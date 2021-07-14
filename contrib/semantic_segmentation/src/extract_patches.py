import argparse
import json
import logging
import multiprocessing
import os
from functools import partial
from os.path import join
from typing import Dict, List
import glob

import numpy as np

from patching.patching import extract_pyramid_patches

from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_patch_and_mask(
    patch_images,
    name,
    output_images_dir,
    patch_folder,
    mask_folder,
    data_folder,
):
    for i in range(len(patch_images)):
        # saving patch
        patch_path = os.path.join(output_images_dir, data_folder, patch_folder)
        patch_name = "_".join([name.split(".")[0], str(i)]) + ".png"
        patch_images[i][0].save(patch_path + "/" + patch_name)

        # saving mask
        mask_path = os.path.join(output_images_dir, data_folder, mask_folder)
        mask_name = "_".join([name.split(".")[0], str(i)]) + ".png"
        patch_images[i][1].save(mask_path + "/" + mask_name)


def process_file(
    tuple_value,
    start,
    end,
    train_indexes,
    val_indexes,
    test_indexes,
    output_images_dir,
    patch_folder,
    mask_folder,
):
    index, _ = tuple_value

    if index >= start and (end == -1 or index <= end):
        image_filepath = join(input_images_dir, all_images[index])
        mask_filepath = join(input_masks_dir, all_masks[index])
        image = Image.open(image_filepath)
        mask = Image.open(mask_filepath)
        name = image_filepath.split("/")[-1]

        patch_images = extract_pyramid_patches(
            image=image, 
            mask=mask,
            classes=classes,                                                               
            patch_dimension=(patch_width, patch_height), 
            pyramid_dimensions=pyramid_patches,
            window_overlap=float(window_overlap),
            threshold=int(threshold))
    
        if index in train_indexes:
            save_patch_and_mask(
                patch_images,
                name,
                output_images_dir,
                patch_folder,
                mask_folder,
                "train",
            )
        if index in val_indexes:
            save_patch_and_mask(
                patch_images,
                name,
                output_images_dir,
                patch_folder,
                mask_folder,
                "validation",
            )
        if index in test_indexes:
            save_patch_and_mask(
                patch_images,
                name,
                output_images_dir,
                patch_folder,
                mask_folder,
                "test",
            )

        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-images-dir", type=str, required=True)
    parser.add_argument("--input-masks-dir", type=str, required=True)
    parser.add_argument("--output-images-dir", type=str, required=True)
    parser.add_argument("--patch_folder", type=str, required=True)
    parser.add_argument("--mask_folder", type=str, required=True)
    parser.add_argument("--window_overlap", type=str, required=True)
    parser.add_argument("--patch_height", type=str, required=True)
    parser.add_argument("--patch_width", type=str, required=True)
    parser.add_argument("--pyramid_patches", type=str, required=True)
    parser.add_argument("--threshold", type=str, required=True)
    parser.add_argument("--classes", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--val_perc", type=str, required=True)
    parser.add_argument("--test_perc", type=str, required=True)

    args = parser.parse_args()

    input_images_dir: str = args.input_images_dir
    input_masks_dir: str = args.input_masks_dir
    output_images_dir: str = args.output_images_dir
    patch_folder: str = args.patch_folder
    mask_folder: str = args.mask_folder
    window_overlap: str = args.window_overlap
    patch_height: int = int(args.patch_height)
    patch_width: int = int(args.patch_width)
    pyramid_patches: List[int] = [int(x) for x in args.pyramid_patches.split(",")]
    threshold: str = args.threshold
    classes: List[int] = [int(x) for x in args.classes.split(",")]
    start: int = int(args.start)
    end: int = int(args.end)
    seed: int = int(args.seed)
    val_perc: float = float(args.val_perc)
    test_perc: float = float(args.test_perc)

    np.random.seed(seed)
    
    all_images = list(glob.iglob(f'{input_images_dir}/*.jpg'))
    all_masks = list(glob.iglob(f'{input_masks_dir}/*.png'))
    all_images_indexes = list(range(len(all_images)))

    np.random.shuffle(all_images_indexes)

    train_indexes, val_indexes, test_indexes = np.split(
        all_images_indexes,
        [
            int(len(all_images_indexes) * val_perc),
            int(len(all_images_indexes) * test_perc),
        ],
    )

    for data_folder in ["train", "validation", "test"]:
        patch_path = os.path.join(output_images_dir, data_folder, patch_folder)
        os.makedirs(patch_path, exist_ok=True)
        print("Directory '%s' created" % patch_path)

        mask_path = os.path.join(output_images_dir, data_folder, mask_folder)
        os.makedirs(mask_path, exist_ok=True)
        print("Directory '%s' created" % mask_path)

    print(f"Running in parallel on {os.cpu_count()} processes")
    pool = multiprocessing.Pool()
    images_processed = list(
        pool.imap(
            partial(
                process_file,
                start=start,
                end=end,
                train_indexes=train_indexes,
                val_indexes=val_indexes,
                test_indexes=test_indexes,
                output_images_dir=output_images_dir,
                patch_folder=patch_folder,
                mask_folder=mask_folder,
            ),
            enumerate(range(len(all_images))),
        )
    )
    pool.close()
    pool.join()

    print(f"Processed {sum(images_processed)} images")
    print(
        f"Train: {len(train_indexes)} Validation: {len(val_indexes)} Test: {len(test_indexes)}"
    )
