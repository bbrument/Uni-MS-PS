import os
import argparse
import logging
import time
from tqdm import tqdm
from utils import load_model
from run import run

# Setup logging
def setup_logging(log_level=logging.INFO):
    """Setup comprehensive logging for inference monitoring"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('inference.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--nb_img", type=int, default=-1, help="number of images")
parser.add_argument("--folder_save", type=str,
                    default='inference', help="path_to_save_results")
parser.add_argument("--path_obj", type=str,
                    required=True, help= 'path_to_imgs')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--calibrated', action='store_true')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
args = parser.parse_args()

# Setup logging
logger = setup_logging(logging.DEBUG if args.verbose else logging.INFO)

logger.info("="*60)
logger.info("PHOTOMETRIC STEREO INFERENCE STARTED")
logger.info("="*60)
logger.info(f"Arguments: {vars(args)}")

mode_inference = True

obj_name = args.path_obj.split(os.sep)[-1]
if len(obj_name)==0:
    obj_name = args.path_obj.split(os.sep)[-2]

logger.info(f"Processing object: {obj_name}")
logger.info(f"Input path: {args.path_obj}")
logger.info(f"Output folder: {args.folder_save}")
logger.info(f"CUDA enabled: {args.cuda}")
logger.info(f"Calibrated mode: {args.calibrated}")

# Load model with timing
logger.info("Loading model...")
start_time = time.time()

with tqdm(desc="Loading model", unit="step") as pbar:
    model = load_model(path_weight="weights",
                       cuda=args.cuda,
                       mode_inference=mode_inference,
                       calibrated=args.calibrated)
    pbar.update(1)

load_time = time.time() - start_time
logger.info(f"Model loaded successfully in {load_time:.2f}s")

# Run inference with timing
logger.info("Starting inference...")
start_time = time.time()

run(model=model,
    path_obj=args.path_obj,
    nb_img=args.nb_img,
    folder_save=args.folder_save,
    obj_name=obj_name,
    calibrated=args.calibrated)

total_time = time.time() - start_time
logger.info(f"Inference completed successfully in {total_time:.2f}s")
logger.info("="*60)