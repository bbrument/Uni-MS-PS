import numpy as np
from PIL import Image
import os
import torch
import cv2
import logging
import time  # Add this import
from tqdm import tqdm
from Transformer_multi_res_7 import Transformer_multi_res_7

logger = logging.getLogger(__name__)

def resize(img, expected_size):
    img = cv2.resize(img,
                     expected_size,
                     interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_with_padding(img, expected_size):
    if type(img)==np.ndarray:
        if img.dtype==np.uint16:
            img1 = img.astype(np.uint8)
        else:
            img1 = img
        img1 = Image.fromarray(img1)
    else:
        img1 = img
    img1.thumbnail((expected_size[0], expected_size[1]))

    delta_width = expected_size[0] - img1.size[0]
    delta_height = expected_size[1] - img1.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width,
               pad_height,
               delta_width - pad_width,
               delta_height - pad_height)
    
    # Convert PIL image back to numpy array to get actual dimensions
    img_array = np.array(img1)
    
    # Handle different channel configurations
    if len(img_array.shape) == 2:
        # Grayscale image
        img_channels = 1
        img2 = np.zeros((expected_size[0], expected_size[1], 3))
        # Convert grayscale to RGB for consistency
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        # RGBA image - remove alpha channel
        img_channels = 3
        img_array = img_array[:, :, :3]  # Keep only RGB channels
        img2 = np.zeros((expected_size[0], expected_size[1], 3))
    else:
        # RGB image
        img_channels = img_array.shape[2]
        img2 = np.zeros((expected_size[0], expected_size[1], img_channels))

    if padding[3]!=0 and padding[2]!=0:
        img2[padding[1]:-padding[3],
             padding[0]:-padding[2]] = img_array
    elif padding[3]!=0:
        img2[padding[1]:-padding[3],
             padding[0]:] = img_array
    elif padding[2]!=0:
        img2[padding[1]:,
             padding[0]:-padding[2]] = img_array
    else:
        img2[padding[1]:,
             padding[0]:] = img_array
    return img2, padding



def depadding(img, padding):
    img = np.array(img)
    if padding[3]!=0 and padding[2]!=0:
        img = img[padding[1]:-padding[3],
                  padding[0]:-padding[2]]
    elif padding[3]!=0:
        img = img[padding[1]:-padding[3],
                  padding[0]:]
    elif padding[2]!=0:
        img = img[padding[1]:,
                  padding[0]:-padding[2]]
    else:
        img = img[padding[1]:,
                  padding[0]:]
    return img

def normal_to_rgb(img):
    return (((img+1)/2)*255).astype(np.uint8)

def normal_to_rgb_16bits(img):
    return (((img+1)/2)*65535).astype(np.uint16)
   

def get_nb_stage(shape):
    max_shape = np.max(shape)
    nb_stage = np.ceil(np.log2(max_shape/32))+1
    nb_stage = int(nb_stage)
    return nb_stage
    
    
def load_imgs_mask(path, 
                   nb_img,
                   calibrated=False,
                   filenames=None,
                   max_size=None):
    
    logger.info(f"Loading images from: {path}")
    
    if filenames is None:
        possible_file = os.listdir(path)
    else:
        possible_file = filenames

    temp = []
    
    # Filter valid image files
    for file in possible_file:
        if ".png" in file and "mask" not in file and "Normal" not in file and "normal" not in file:
            temp.append(file)
        elif ".jpg" in file and "mask" not in file and "Normal" not in file and "normal" not in file:
            temp.append(file)
        elif ".TIF" in file and "mask" not in file and "Normal" not in file and "normal" not in file:
            temp.append(file)
        elif ".JPG" in file and "mask" not in file and "Normal" not in file and "normal" not in file:
            temp.append(file)
    
    logger.info(f"Found {len(temp)} valid image files")
    
    # Load mask
    file_mask = os.path.join(path, "mask.png")
    if os.path.exists(file_mask):
        logger.info("Loading mask file")
        mask = cv2.imread(file_mask)
        # Handle RGBA masks by converting to RGB
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
            logger.debug("Converted RGBA mask to RGB")
    else:
        logger.info("No mask file found, creating default mask")
        file_img_example = os.path.join(path, temp[0])
        img_example = cv2.imread(file_img_example)
        # Handle RGBA example images
        if len(img_example.shape) == 3 and img_example.shape[2] == 4:
            img_example = img_example[:, :, :3]
        mask = np.ones(img_example.shape, 
                       dtype=np.uint8)
    
    # Check if we need to transpose (portrait to landscape)
    is_portrait = mask.shape[0] > mask.shape[1]  # height > width
    if is_portrait:
        logger.info("Portrait orientation detected, transposing to landscape")
        mask = np.transpose(mask, (1, 0, 2))  # Transpose height and width
    
    if max_size is not None:
        if mask.shape[0]>max_size or mask.shape[1]>max_size:
            logger.info(f"Resizing mask to max_size: {max_size}")
            mask = cv2.resize(mask,
                              (max_size, max_size))
            
    original_shape = mask.shape
    logger.info(f"Mask shape: {original_shape}")
    
    coord = np.argwhere(mask[:,:,0]>0)
    x_min, x_max = np.min(coord[:,0]), np.max(coord[:,0])
    y_min, y_max = np.min(coord[:,1]), np.max(coord[:,1])

    x_max_pad = mask.shape[0] - x_max
    y_max_pad = mask.shape[1] - y_max
        
    mask = mask[x_min:x_max,
                y_min:y_max]
        
    nb_stage = get_nb_stage(mask.shape)
    size_img = 32*2**(nb_stage-1)
    
    logger.info(f"Number of processing stages: {nb_stage}")
    logger.info(f"Target image size: {size_img}x{size_img}")
    
    mask, _ = resize_with_padding(mask,
                                      expected_size=(size_img,
                                                     size_img))
    mask = (mask>0)
    mask = mask[:,:,0]
    
    imgs = []
    
    if nb_img is None or nb_img>=len(temp) or nb_img==-1:
        files = np.array(temp)
    else:
        files = np.random.choice(temp, nb_img, replace=False)
    
    logger.info(f"Processing {len(files)} images")
    
    # Process images with progress bar
    for file in tqdm(files, desc="Loading images", unit="img"):
        file1 = os.path.join(path, file)
        img = cv2.imread(file1, 
                         cv2.IMREAD_UNCHANGED)
        
        # Handle different image formats
        if len(img.shape)==2:
            # Grayscale image
            img = np.expand_dims(img, -1)
            img = np.concatenate((img, img, img),
                                 axis=-1)
            logger.debug(f"Converted grayscale image: {file}")
        elif len(img.shape)==3 and img.shape[2]==4:
            # RGBA image - remove alpha channel
            img = img[:, :, :3]
            logger.debug(f"Converted RGBA image: {file}")
        elif len(img.shape)==3 and img.shape[2]==1:
            # Single channel image
            img = np.concatenate((img, img, img),
                                 axis=-1)
            logger.debug(f"Converted single channel image: {file}")
        
        # Transpose image if we detected portrait orientation
        if is_portrait:
            img = np.transpose(img, (1, 0, 2))  # Transpose height and width
        
        if max_size is not None:
            if img.shape[0]>max_size or img.shape[1]>max_size:
                img = cv2.resize(img,
                                 (max_size, max_size))
                
        img = img[x_min:x_max,
                  y_min:y_max]
            
        img, padding = resize_with_padding(img=img,
                                           expected_size=(size_img, size_img))

        img = img.astype(np.float32)
        mean_img = np.mean(img, -1)
        mean_img = mean_img.flatten()
        mean_img1 = np.mean(mean_img[mask.flatten()])
        img = img/mean_img1
        
        imgs.append(img)
    
    imgs = np.array(imgs)
    imgs = np.moveaxis(imgs,
                       -1,
                       0)
    imgs = torch.from_numpy(imgs).unsqueeze(0).float()
    
    if calibrated:
        logger.info("Loading light directions for calibrated mode")
        dirs_file = os.path.join(path,
                                 "light_directions.txt")
        dirs_all = np.loadtxt(dirs_file)
        dirs = []
        for key in files:
            key = int(key.split(".")[0])-1
            d = dirs_all[key]
            dirs.append([d[0],
                         d[1],
                         d[2]])
        
        dirs = np.array(dirs)
        dirs = torch.from_numpy(dirs).movedim(1,0).unsqueeze(0)
        dirs.unsqueeze_(-1).unsqueeze_(-1)
        
        dirs = dirs.expand_as(imgs)[:,:,:,:]
    
        imgs = torch.cat([imgs, dirs], 1).float()
    
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    
    logger.info(f"Image tensor shape: {imgs.shape}")
    logger.info(f"Mask tensor shape: {mask.shape}")
    
    return imgs, mask, padding, [x_min, x_max_pad, y_min, y_max_pad], original_shape, is_portrait


def load_model(path_weight, cuda,
               calibrated, mode_inference=False): 
    logger.info("Initializing model...")
    
    if calibrated:
        file_weight = os.path.join(path_weight, "model_calibrated.pth")
        logger.info("Using calibrated model")
    else:
        file_weight = os.path.join(path_weight, "model_uncalibrated.pth")
        logger.info("Using uncalibrated model")
    
    if not os.path.exists(file_weight):
        raise FileNotFoundError(f"Model weights not found at: {file_weight}")
    
    if calibrated:
        # Optimize batch sizes for better GPU utilization
        model = Transformer_multi_res_7(c_in=6,
                                        batch_size_encoder=6,  # Increased
                                        batch_size_transformer=8000)  # Increased
    else:
        model = Transformer_multi_res_7(c_in=3,
                                        batch_size_encoder=6,  # Increased
                                        batch_size_transformer=8000)  # Increased
    
    logger.info(f"Loading weights from: {file_weight}")
    model.load_weights(file=file_weight)
    model.eval()
    
    if mode_inference:
        logger.info(f"Setting inference mode (CUDA: {cuda})")
        model.set_inference_mode(use_cuda_eval_mode=cuda)
        # Enable memory optimizations
        if hasattr(model, 'use_pinned_memory'):
            model.use_pinned_memory = True
            logger.debug("Enabled pinned memory optimization")
    elif cuda:
        logger.info("Moving model to CUDA")
        model.cuda()
    
    # Log model configuration
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


def process_normal(model, imgs, mask):
    logger.info("Starting normal estimation...")
    
    nb_stage = get_nb_stage(mask.shape)
    logger.info(f"Processing with {nb_stage} stages")
    
    x = {}
    x["imgs"] = imgs
    x["mask"] = mask

    start_time = time.time()
    with torch.no_grad():
        a = model.process(x, nb_stage)
        normal = a["n"].squeeze().movedim(0,-1).numpy()
    
    process_time = time.time() - start_time
    logger.info(f"Normal estimation completed in {process_time:.2f}s")
    logger.info(f"Output normal shape: {normal.shape}")
    
    return normal

