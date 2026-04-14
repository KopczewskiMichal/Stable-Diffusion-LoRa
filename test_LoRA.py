import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os


# Ścieżki
# OUTPUT_DIR = "./Bird_LoRA"
OUTPUT_DIR = "./Bird_LoRA_no_output"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
# PROMPT = "a professional studio macro photo of skp_red_bird figurine, dramatic cinematic lighting, bokeh, 8k resolution, photorealistic, sharp focus on the face, matte plastic texture"
# PROMPT = "a photo of skp_red_bird figurine, standing in a level from Angry Birds game, wooden crates in background, hyperrealistic"
# PROMPT = "a photo of bird figurine, from angry birds game,underwater, at the bottom of the ocean, hyperrealistic"
# PROMPT = "a photo of skp_red_bird figurine, like in angry birds game,underwater, at the bottom of the ocean, hyperrealistic"
PROMPT = "a photo of skp_red_bird figurine as a flash drive, office style, computer behind, hyperrealistic"
# PROMPT = "a photo of skp_red_bird figurine, like in angry birds game, underwater, at the bottom of the ocean, hyperrealistic"
# PROMPT = "photo of a flash drive in shape of skp_red_bird, plugged into laptop, office style, hyperrealistic"
# PROMPT = "a photo of angry bird in game style"


print(">>> Ładowanie modelu bazowego...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    safety_checker=None
).to("cuda")

print(f">>> Próba załadowania LoRA z: {OUTPUT_DIR}")

# FIX: Ładujemy LoRA bezpośrednio przez bibliotekę PEFT (tak jak była trenowana)
# Zamiast polegać na pipe.load_lora_weights, która szuka innych nazw plików
if os.path.exists(os.path.join(OUTPUT_DIR, "adapter_model.bin")) or os.path.exists(
        os.path.join(OUTPUT_DIR, "adapter_model.safetensors")):
    try:
        # Owijamy UNeta w PeftModel i ładujemy wagi
        pipe.unet = PeftModel.from_pretrained(pipe.unet, OUTPUT_DIR)

        # Scalamy wagi dla lepszej wydajności (opcjonalne, ale zalecane)
        pipe.unet = pipe.unet.merge_and_unload()

        print(">>> SUKCES! LoRA załadowana i scalona.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit()
else:
    print(f"BŁĄD: W folderze {OUTPUT_DIR} brakuje pliku adapter_model.bin/.safetensors!")
    print("Czy trening na pewno się zakończył?")
    exit()

# Generowanie
print(">>> Generowanie...")
for i in range(6):
    image = pipe(PROMPT, num_inference_steps=60, guidance_scale=7.5, negative_prompt="deformed, blurred").images[0]
    image.save(f"generated/bird_{i}_pendrive.png")
print(">>> Gotowe: zapisano zdjęcia")