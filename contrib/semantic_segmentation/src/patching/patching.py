from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

def extract_patches(
    image: object,
    mask: object,
    classes: List[int],
    patch_dimension: Tuple[int, int] = (1000, 1000),
    window_overlap: float = 0.1,
    threshold: int = 100,
) -> List[Tuple[object, object]]:
    """For an input image with a relative mask return
    a list of all extracted patched images and corresponding mask images

    Parameters
    ----------
    image : object
        Original source image
    mask : object
        Original mask image
    classes : List[int]
        list of classes to use for the patch image
    patch_dimension : Tuple(Int, Int)
        Width and Height of the extracted patch
    window_overlap : Float
        increment window by % of patch_dimension
    threshold : Int
        minimum number of pixels in patch mask

    Returns
    -------
    patch_list : List[Tuple[object, object]]
        List of all extracted patches and corresponding mask images
    """
    patches = []
    width = image.width
    height = image.height

    mask_array = np.asarray(mask, dtype=np.uint8)
    mask_array = np.where(mask_array[:,:,0] == 255, 1, 0).astype(np.uint8)
    mask_image = Image.fromarray(mask_array)
    # Get monochromatic mask array in order to count number of pixels different than background.
    # This array must also be transposed due to differences in the x,y coordinates between
    # Pillow and Numpy matrix
    mask_mono_array = np.where(mask_array > 0, 1, 0).astype("uint8").transpose()

    processed = set()

    # move window of patch_dimension on the original image
    for x in range(0, width, int(patch_dimension[0] * window_overlap)):
        for y in range(0, height, int(patch_dimension[1] * window_overlap)):
            # get patch dimension
            x = min(x, width - patch_dimension[0])
            y = min(y, height - patch_dimension[1])

            if (x, y) not in processed:
                processed.add((x, y))
                if (
                    mask_mono_array[
                        x : x + patch_dimension[0], y : y + patch_dimension[1]
                    ].sum()
                    >= threshold
                ):
                    patch_pos = (x, y, x + patch_dimension[0], y + patch_dimension[1])
                    patch_image = image.crop(patch_pos)
                    patch_mask_image = mask_image.crop(patch_pos)

                    patches.append((patch_image, patch_mask_image))

    return patches

def extract_pyramid_patches(
    image: object,
    mask: object,
    classes: List[int],
    patch_dimension: Tuple[int, int] = (1000, 1000),
    window_overlap: float = 0.1,
    threshold: int = 100,
) -> List[Tuple[object, object]]:
    """For an input image with a relative mask return
    a list of all extracted patched images and corresponding mask images

    Parameters
    ----------
    image : object
        Original source image
    mask : object
        Original mask image
    classes : List[int]
        list of classes to use for the patch image
    patch_dimension : Tuple(Int, Int)
        Width and Height of the extracted patch
    window_overlap : Float
        increment window by % of patch_dimension
    threshold : Int
        minimum number of pixels in patch mask

    Returns
    -------
    patch_list : List[Tuple[object, object]]
        List of all extracted patches and corresponding mask images
    """
    patch_images = []

    patches = extract_patches(
        image=image, 
        mask=mask,
        classes=classes,                                                               
        patch_dimension=patch_dimension, 
        window_overlap=window_overlap,
        threshold=threshold)

    patch_images.extend((i.resize(patch_dimension), p.resize(patch_dimension)) for i, p in patches)

    return patch_images

