import torch
import os

from diffusers import StableDiffusionPipeline

# ================= KONFIGURACJA =================
# Ścieżka do pliku wynikowego z treningu
EMBEDDING_PATH = "./Bird_TI_8bit/learned_embeds.bin"

# Twój token (musi być taki sam jak w treningu!)
TOKEN = "<skp_red_bird>"
# Prompt testowy - sprawdzamy czy model umie wstawić obiekt w nowe tło
PROMPT = f"a photo of <skp_red_bird> like in angry birds game, underwater, at the bottom of the ocean, hyperrealistic"

NEGATIVE_PROMPT = "blur, low quality, distortion, ugly, bad anatomy"


# ================================================

def generate_images(n=1):
    if not os.path.exists(EMBEDDING_PATH):
        print(f"[BŁĄD] Nie znaleziono pliku: {EMBEDDING_PATH}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Uruchamiam na: {device}")

    print("-> Ładowanie modelu bazowego...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,  # Ważne dla 8GB VRAM
        use_safetensors=True
    ).to(device)

    print(f"-> Wstrzykiwanie embeddingu: {EMBEDDING_PATH}")
    try:
        # To jest ta magiczna linijka, która dodaje Twój "wyraz" do słownika modelu
        pipe.load_textual_inversion(EMBEDDING_PATH)
    except Exception as e:
        print(f"[BŁĄD] Nie udało się załadować embeddingu: {e}")
        return

    print(f"-> Generowanie: '{PROMPT}'")

    for i in range (n):
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=50,  # 30-50 to standard
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]

        image.save(f"generated/bird_TI_{i}.jpg")


if __name__ == "__main__":
    generate_images(6)