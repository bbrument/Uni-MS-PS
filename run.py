import cv2
import os
import torch
import scipy.io
import numpy as np
import logging
import time
from tqdm import tqdm
from utils import depadding, normal_to_rgb, normal_to_rgb_16bits
from utils import load_imgs_mask, process_normal

logger = logging.getLogger(__name__)

def run(model, path_obj, nb_img, folder_save, obj_name, calibrated):
    logger.info(f"Starting processing for object: {obj_name}")
    
    # Stage 1: Load images and mask
    logger.info("Stage 1/4: Loading images and mask")
    start_time = time.time()
    
    imgs, mask, padding, zoom_coord, original_shape = load_imgs_mask(path=path_obj,
                                                                     nb_img=nb_img,
                                                                     calibrated=calibrated)
    
    load_time = time.time() - start_time
    logger.info(f"Images loaded in {load_time:.2f}s")
    
    # Stage 2: Process normal estimation
    logger.info("Stage 2/4: Estimating normals")
    start_time = time.time()
    
    normal = process_normal(model=model,
                            imgs=imgs,
                            mask=mask)
    
    process_time = time.time() - start_time
    logger.info(f"Normal estimation completed in {process_time:.2f}s")
    
    # Stage 3: Post-processing
    logger.info("Stage 3/4: Post-processing results")
    start_time = time.time()
    
    with tqdm(desc="Post-processing", total=4, unit="step") as pbar:
        # Remove padding
        normal_resize = depadding(normal, padding=padding)
        pbar.update(1)
        
        # Normalize
        normal_resize = torch.from_numpy(normal_resize)
        normal_resize = torch.nn.functional.normalize(normal_resize, 2, -1).numpy()
        pbar.update(1)
        
        # Add padding back to original size
        pad_x_min = np.zeros((zoom_coord[0], normal_resize.shape[1], 3))
        pad_x_max = np.zeros((zoom_coord[1], normal_resize.shape[1], 3))
        normal_resize = np.concatenate((pad_x_min,
                                        normal_resize,
                                        pad_x_max), axis=0)
        pbar.update(1)
                
        pad_y_min = np.zeros((normal_resize.shape[0], zoom_coord[2], 3))
        pad_y_max = np.zeros((normal_resize.shape[0], zoom_coord[3], 3))
                
        normal_resize = np.concatenate((pad_y_min,
                                        normal_resize,
                                        pad_y_max), axis=1)
        pbar.update(1)
    
    postprocess_time = time.time() - start_time
    logger.info(f"Post-processing completed in {postprocess_time:.2f}s")
    
    # Stage 4: Save results
    logger.info("Stage 4/4: Saving results")
    start_time = time.time()
    
    normal_resize_rgb = normal_to_rgb_16bits(normal_resize)

    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
        logger.info(f"Created output directory: {folder_save}")
    
    # Save files with progress tracking
    with tqdm(desc="Saving files", total=2, unit="file") as pbar:
        # Save PNG
        png_path = os.path.join(folder_save, "{}.png".format(obj_name))
        cv2.imwrite(png_path, normal_resize_rgb[:,:,::-1])
        logger.info(f"Saved PNG: {png_path}")
        pbar.update(1)
        
        # Save MAT
        mat_path = os.path.join(folder_save, "{}.mat".format(obj_name))
        scipy.io.savemat(mat_path, {'Normal_est': normal_resize})
        logger.info(f"Saved MAT: {mat_path}")
        pbar.update(1)
    
    save_time = time.time() - start_time
    logger.info(f"Results saved in {save_time:.2f}s")
    
    # Summary
    total_time = load_time + process_time + postprocess_time + save_time
    logger.info("="*50)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*50)
    logger.info(f"Object: {obj_name}")
    logger.info(f"Input shape: {original_shape}")
    logger.info(f"Output shape: {normal_resize.shape}")
    logger.info(f"Load time: {load_time:.2f}s")
    logger.info(f"Process time: {process_time:.2f}s") 
    logger.info(f"Post-process time: {postprocess_time:.2f}s")
    logger.info(f"Save time: {save_time:.2f}s")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info("="*50)