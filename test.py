"""
Evaluation Script
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import models

import numpy as np

from utils.config import Config
from src.dataset import Sentinel
from src.pix2pix import Pix2Pix
from src.metric import extract_features, calculate_fid
import logging

# import ssl
# import urllib.request

# ssl._create_default_https_context = ssl._create_unverified_context


# Configure logging
# logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load configuration
        print("Loading configuration...")
        config = Config('config.yaml')
        print("Configuration loaded successfully.")

        # Set device
        device = torch.device(config['training']['device'])
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Load inception model
        print("Loading Inception model...")
        inception = models.inception_v3(weights='DEFAULT', transform_input=False).eval().to(device)
        print("Inception model loaded.")

        # Transforms
        print("Setting up transforms...")
        transform = v2.Compose([
            v2.Resize(342),
            v2.CenterCrop(299),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Transforms set up.")

        # Load dataset
        print("Loading dataset...")
        dataset = Sentinel(
            root_dir=config['dataset']['root_dir'],
            split_type="test",
            split_mode=config['dataset']['split_mode'],
            split_ratio=config['dataset']['split_ratio'],
            split_file=config['dataset']['split_file'],
            seed=config['dataset']['seed']
        )
        print(f"Dataset loaded with {len(dataset)} samples.")

        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=config['dataset']['shuffle'],
            num_workers=config['training']['num_workers']
        )
        print("Dataloader initialized.")

        # Create model
        print("Initializing Pix2Pix model...")
        model = Pix2Pix(
            c_in=config['model']['c_in'],
            c_out=config['model']['c_out'],
            is_train=False,
            use_upsampling=config['model']['use_upsampling'],
            mode=config['model']['mode'],
        ).to(device).eval()
        print("Pix2Pix model initialized.")

        # Load generator checkpoint
        gen_checkpoint = Path(config['training']['gen_checkpoint'])
        if not gen_checkpoint.exists():
            raise FileNotFoundError(f"Generator checkpoint file not found: {gen_checkpoint}\nPlease check config.yaml")
        print("Loading model checkpoint...")
        model.load_model(gen_path=gen_checkpoint)
        print("Model checkpoint loaded.")

        target_features = []
        fake_features = []

        print("Starting evaluation loop...")
        for idx, (real_images, target_images) in enumerate(dataloader):
            print(f"Processing batch {idx + 1}")
            real_images, target_images = real_images.to(device), target_images.to(device)

            # Pix2Pix.generate() gets a scaled tensor ([0,1]) returns a uint8 tensor ([0,255])
            fake_images = model.generate(real_images, is_scaled=True, to_uint8=True)
            print(f"Generated fake images for batch {idx + 1}.")

            # Get target features
            target_images = (target_images * 255).to(dtype=torch.uint8)
            target_images = transform(target_images)
            target_feats = extract_features(target_images, inception)
            target_features.append(target_feats.cpu().numpy())

            # Get fake features
            fake_images = transform(fake_images)
            fake_feats = extract_features(fake_images, inception)
            fake_features.append(fake_feats.cpu().numpy())
            print(f"Features extracted for batch {idx + 1}.")

        # Convert lists to numpy arrays
        print("Converting features to numpy arrays...")
        real_features = np.concatenate(target_features, axis=0)
        generated_features = np.concatenate(fake_features, axis=0)

        # Compute FID score
        print("Calculating FID score...")
        fid_score = calculate_fid(real_features, generated_features)
        print(f"FID Score: {fid_score}")
        print(f"FID Score calculated: {fid_score}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()


