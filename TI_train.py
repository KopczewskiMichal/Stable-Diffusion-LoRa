import os
import torch
import glob
import bitsandbytes as bnb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from torch.cuda.amp import GradScaler

# --- KONFIGURACJA NITRO ---
PATH_TO_IMAGES = r"./Data/Bird"
OUTPUT_DIR = "./Bird_TI_Nitro"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

PLACEHOLDER_TOKEN = "<skp_red_bird>"
INITIALIZER_TOKEN = "red"

STEPS = 3000
BATCH_SIZE = 4  # Przy pre-caching możesz dać większy batch na 8GB!
LEARNING_RATE = 5e-4

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TEMPLATES = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a close-up photo of a {}",
    "a macro shot of {}"
]


# --- 1. PRZYGOTOWANIE DANYCH (CACHE) ---
def prepare_latents_cache(folder, vae, tokenizer, size=512):
    print(">>> Generowanie cache (Latents + Flips)...")

    # Znajdź zdjęcia
    images_paths = []
    exts = ('*.png', '*.jpg', '*.jpeg', '*.webp')
    for ext in exts:
        images_paths.extend(glob.glob(os.path.join(folder, ext)))
        images_paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    images_paths = sorted(list(set(images_paths)))

    # Transformacje (Tylko resize i crop)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    cached_data = []
    vae.to("cuda")

    for path in set(images_paths):
        try:
            img = Image.open(path).convert("RGB")

            # Wersja 1: Oryginał
            pixel_values = transform(img).unsqueeze(0).to("cuda")
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            cached_data.append(latents.cpu())  # Zrzut do RAM

            # Wersja 2: Flip (Darmowa augmentacja)
            img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
            pixel_values_flip = transform(img_flip).unsqueeze(0).to("cuda")
            with torch.no_grad():
                latents_flip = vae.encode(pixel_values_flip).latent_dist.sample() * 0.18215
            cached_data.append(latents_flip.cpu())

        except Exception as e:
            print(f"Błąd przy {path}: {e}")

    print(f">>> Zcache'owano {len(cached_data)} tensorów (oryginały + odbicia).")
    return cached_data


class LatentsDataset(Dataset):
    def __init__(self, latents, templates, placeholder_token):
        self.latents = latents
        self.templates = templates
        self.placeholder_token = placeholder_token

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        # Losowy szablon tekstu
        text = self.templates[i % len(self.templates)].format(self.placeholder_token)
        return {"latents": self.latents[i], "text": text}


# --- 2. SETUP MODELU ---
print(">>> Ładowanie modeli...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to("cuda")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to("cuda")  # Potrzebny tylko na chwilę
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.float16).to("cuda")

# Dodanie tokenu
num_added_tokens = tokenizer.add_tokens(PLACEHOLDER_TOKEN)
if num_added_tokens == 0:
    raise ValueError(f"Token {PLACEHOLDER_TOKEN} już istnieje!")

text_encoder.resize_token_embeddings(len(tokenizer))
token_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
initializer_token_id = tokenizer.encode(INITIALIZER_TOKEN, add_special_tokens=False)[0]

print(f">>> Inicjalizacja: {PLACEHOLDER_TOKEN} jak '{INITIALIZER_TOKEN}'")
with torch.no_grad():
    text_encoder.get_input_embeddings().weight[token_id] = \
        text_encoder.get_input_embeddings().weight[initializer_token_id].clone()

# --- 3. PRE-CACHING I CZYSZCZENIE ---
latents_cache = prepare_latents_cache(PATH_TO_IMAGES, vae, tokenizer)

# Wywalamy VAE - nie jest już potrzebny!
del vae
torch.cuda.empty_cache()
print(">>> VAE usunięty z pamięci. Lecimy z treningiem.")

# Dataset i DataLoader (num_workers=0 dla Windowsa!)
dataset = LatentsDataset(latents_cache, TEMPLATES, PLACEHOLDER_TOKEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Zamrażanie
unet.requires_grad_(False)
text_encoder.text_model.encoder.requires_grad_(False)
text_encoder.text_model.final_layer_norm.requires_grad_(False)
text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
text_encoder.get_input_embeddings().requires_grad_(True)

optimizer = bnb.optim.AdamW8bit(text_encoder.get_input_embeddings().parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# --- 4. SZYBKA PĘTLA TRENINGOWA ---
print(f">>> START NITRO (Steps: {STEPS}, Batch: {BATCH_SIZE})")
global_step = 0
data_iter = iter(dataloader)

while global_step < STEPS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    # Latenty już gotowe, wchodzą prosto na GPU
    latents = batch["latents"].squeeze(1).to("cuda", dtype=torch.float16)

    texts = batch["text"]
    input_ids = tokenizer(
        texts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
    ).input_ids.to("cuda")

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device="cuda").long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Autocast + FP16 UNet = Prędkość
    with torch.autocast("cuda"):
        encoder_hidden_states = text_encoder(input_ids)[0]
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(dtype=torch.float16)).sample
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())

    scaler.scale(loss).backward()

    # Update co krok (nie trzeba Gradient Accumulation bo mamy większy Batch Size dzięki usunięciu VAE)
    scaler.unscale_(optimizer)
    grads = text_encoder.get_input_embeddings().weight.grad
    index_grads_to_zero = torch.arange(len(tokenizer)) != token_id
    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    global_step += 1

    if global_step % 200 == 0:
        print(f"Step {global_step}/{STEPS} | Loss: {loss.item():.4f}")

# --- ZAPIS ---
learned_embeds = text_encoder.get_input_embeddings().weight[token_id]
learned_embeds_dict = {PLACEHOLDER_TOKEN: learned_embeds.detach().cpu()}
torch.save(learned_embeds_dict, os.path.join(OUTPUT_DIR, "learned_embeds.bin"))
print(f"Gotowe! Wynik: {os.path.join(OUTPUT_DIR, 'learned_embeds.bin')}")