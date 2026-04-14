import os
import gc
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from itertools import cycle

PATH_TO_IMAGES = r"./Data/Bird"

OUTPUT_DIR = "./Bird_LoRA_no_output"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

INSTANCE_PROMPT = "a photo of skp_red_bird figurine"
CLASS_PROMPT = "a photo of small plastic figurine"

RESOLUTION = 512
BATCH_SIZE = 1
STEPS = 1000
LEARNING_RATE = 1e-4
RANK = 128
ALPHA = 128


class DreamBoothDataset(Dataset):
    def __init__(self, folder, tokenizer, size=512):
        if not os.path.exists(folder):
            raise ValueError(f"Kurczę, folder nie istnieje: {folder}")

        exts = ('*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.JPG')
        self.images = []
        for ext in exts:
            self.images.extend(glob.glob(os.path.join(folder, ext)))
            self.images.extend(glob.glob(os.path.join(folder, ext.upper())))

        self.images = sorted(list(set(self.images)))
        print(f"DEBUG: Znaleziono {len(self.images)} zdjęć w folderze.")

        if len(self.images) == 0:
            raise ValueError("Folder jest pusty! Wrzuć tam jakieś zdjęcia.")

        self.tokenizer = tokenizer
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        try:
            image = Image.open(self.images[i]).convert("RGB")
        except Exception as e:
            print(f"Błąd pliku {self.images[i]}: {e}")
            return self.__getitem__((i + 1) % len(self.images))

        pixel_values = self.transforms(image)

        # Tokenizacja promptu
        tokenized = self.tokenizer(
            INSTANCE_PROMPT,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": tokenized}


print(">>> Ładowanie modeli (VAE, Tokenizer, UNet)...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to("cpu", dtype=torch.float32)
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.float32).to("cuda")
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

print(f">>> Konfiguracja LoRA (Rank: {RANK})...")
lora_config = LoraConfig(
    r=RANK, lora_alpha=ALPHA,
    # target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.05, bias="none"
)
unet = get_peft_model(unet, lora_config)

optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_training_steps=STEPS, num_warmup_steps=100)

dataset = DreamBoothDataset(PATH_TO_IMAGES, tokenizer, RESOLUTION)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(">>> Cache'owanie danych do VRAM (To uwolni CPU i przyspieszy trening)...")
cached_latents = []

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

for batch in dataloader:
    pixel_values = batch["pixel_values"].to("cpu", dtype=torch.float32)
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
    latents = latents.to("cuda", dtype=torch.float16)

    input_ids = batch["input_ids"].to("cuda")
    with torch.no_grad():
        encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=torch.float16)

    cached_latents.append((latents, encoder_hidden_states))

print(f">>> Zcache'owano {len(cached_latents)} batchy. Usuwam zbędne modele...")

del vae
del text_encoder
del tokenizer
del dataset
del dataloader
gc.collect()
torch.cuda.empty_cache()

print(">>> START TRENINGU...")
unet.enable_gradient_checkpointing()
unet.train()
train_iter = cycle(cached_latents)

for step in range(STEPS):
    latents, encoder_hidden_states = next(train_iter)

    with torch.amp.autocast('cuda'):
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device="cuda").long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Krok {step}/{STEPS} | Loss: {loss.item():.4f}")

print(">>> Trening zakończony. Zapisywanie...")
unet.save_pretrained(OUTPUT_DIR)

