import torch
import os
import gc
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb
from itertools import cycle
# --- KONFIGURACJA ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"
INSTANCE_DIR = "./Cactus"
OUTPUT_DIR = "./lora_weights"
PROMPT = "a photo of orange nm_mk_cactus cactus in pot"
RESOLUTION = 512
BATCH_SIZE = 1
STEPS = 1000
