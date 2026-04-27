import os
import torch
import logging
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Configure logging for failed detections
logging.basicConfig(
    #filename='mtcnn_failed_detections.log',
    filename = os.path.join('outputs', 'results', 'mtcnn_failed_detections.log'),
    level=logging.WARNING,
    format='%(asctime)s - FAILED DETECTION: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def preprocess_and_split(raw_dir="data/raw/lfw", processed_dir="data/processed"):
    """
    Detects faces using MTCNN, crops them with a margin, and saves resized 
    versions into separate folders for EfficientNetV2 S, M, and L variants.
    """
    # 1. Device Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing MTCNN on device: {device}")

    # 2. Initialize MTCNN
    # margin=20: Preserves the jawline/forehead for ArcFace
    mtcnn = MTCNN(margin=20, keep_all=False, post_process=False, device=device)

    # 3. Define EfficientNetV2 target resolutions
    variant_sizes = {
        's': 300,  # EfficientNetV2-Small
        'm': 384,  # EfficientNetV2-Medium
        'l': 480   # EfficientNetV2-Large
    }

    # Verify raw data exists
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print(f"[ERROR] Raw data directory '{raw_dir}' is empty or missing.")
        return

    # Count total identities and images for the progress bar
    identities = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    total_images = sum(len(os.listdir(os.path.join(raw_dir, d))) for d in identities)
    print(f"[INFO] Found {len(identities)} identities and {total_images} total images.")

    # 4. Processing Loop
    with tqdm(total=total_images, desc="Processing & Splitting Faces") as pbar:
        for identity_name in identities:
            identity_path = os.path.join(raw_dir, identity_name)
            
            # Create subdirectories for this identity inside s/, m/, and l/
            for variant in variant_sizes.keys():
                os.makedirs(os.path.join(processed_dir, variant, identity_name), exist_ok=True)

            for image_name in os.listdir(identity_path):
                img_path = os.path.join(identity_path, image_name)
                
                try:
                    # Load the raw image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Detect and extract the face (returns a cropped PIL Image or None)
                    face = mtcnn(img)
                    
                    if face is None:
                        # Log the failure and skip to the next image
                        logging.warning(img_path)
                        pbar.update(1)
                        continue
                    
                    # Convert the tensor back to a PIL image for resizing
                    # (MTCNN returns a normalized tensor by default if post_process is not perfectly handled)
                    # For safety, we use the mtcnn.detect to get bounding boxes and crop manually if needed, 
                    # but facenet-pytorch mtcnn(img, save_path=...) handles saving natively.
                    # Since we need multiple sizes, we extract the face tensor, convert it, and resize.
                    
                    # Ensure face is a PIL Image for high-quality Lanczos resizing
                    if isinstance(face, torch.Tensor):
                        # Permute and denormalize back to 0-255 PIL Image
                        face_np = face.permute(1, 2, 0).numpy()
                        face_pil = Image.fromarray(face_np.astype('uint8'))
                    else:
                        face_pil = face

                    # Resize and save for each variant (S, M, L)
                    for variant, size in variant_sizes.items():
                        out_path = os.path.join(processed_dir, variant, identity_name, image_name)
                        resized_face = face_pil.resize((size, size), Image.Resampling.LANCZOS)
                        resized_face.save(out_path)
                        
                except Exception as e:
                    logging.warning(f"{img_path} | ERROR: {str(e)}")
                
                pbar.update(1)

    print(f"\n[INFO] Preprocessing complete! Faces saved to: {processed_dir}")
    print("[INFO] Check 'mtcnn_failed_detections.log' for skipped images.")

if __name__ == '__main__':
    preprocess_and_split()