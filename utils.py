from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

import cv2
import numpy as np
from scipy import stats
from ultralytics import YOLO


def get_rooms(image: np.ndarray, model: YOLO, area_threshold=0.0005) -> List[List[List[int]]]:
    # mask area
    area = image.shape[0] * image.shape[1]
    
    # inference results
    result = model.predict(image)[0]
    
    # Extract masks from results
    masks = result.masks.data.cpu().numpy()
    
    # rooms only masks
    masks = [
        mask for mask, obb in zip(masks, result.boxes)
        if int(obb.cls) == 3
    ]
    
    # resize masks to original image size
    masks = [
        cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]
    
    # filter out overlapping masks
    masks = filter_overlapping_masks(masks)
    
    print('Mask count:', len(masks))
    for i, mask in enumerate(masks):
        cv2.imwrite(f'mask_{i}.png', mask)
    
    # find contours
    room_contours = []
    for mask in masks:
        mask = (mask > 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            room_contours.append(contours[0])
    

    # # threshold
    # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # # invert mask
    # mask_inv = 255 - mask

    # # find contours
    # contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # filter out small contours
    # room_contours = filter(
    #     lambda cnt: cv2.contourArea(cnt) > area * area_threshold,
    #     contours
    # )

    # # sort contours
    # room_contours = sorted(room_contours, key=lambda cnt: cv2.contourArea(cnt))

    # # remove the biggest contour
    # room_contours = filter_contours(room_contours)

    return room_contours

def calculate_overlap(mask1, mask2):
    # Ensure the masks are binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    
    # Calculate intersection
    intersection = np.logical_and(mask1, mask2).astype(np.uint8)
    intersection_area = np.sum(intersection)
    
    # Calculate individual areas
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    
    return intersection_area, area1, area2

# Example function to filter overlapping masks
def filter_overlapping_masks(masks, threshold=0.2):
    num_masks = len(masks)
    keep_masks = np.ones(num_masks, dtype=bool)  # Keep all masks initially

    for i in range(num_masks):
        for j in range(i + 1, num_masks):
            if not keep_masks[i] or not keep_masks[j]:
                continue
            
            intersection_area, area1, area2 = calculate_overlap(masks[i], masks[j])
            min_area = min(area1, area2)
            
            if intersection_area > min_area * threshold:
                # If the intersection area is greater than the smallest area, remove the smaller mask
                if area1 < area2:
                    keep_masks[i] = False
                else:
                    keep_masks[j] = False

    filtered_masks = [masks[i] for i in range(num_masks) if keep_masks[i]]
    return filtered_masks