"""
Unified vision tester for your local Qwen checkpoints on a single RTX 3060.
Swap models with one CLI arg; load mode is auto-set per model. Reports tokens/sec
so you can measure the speed difference yourself.

Setup (once):
    pip install -U "transformers>=4.58" accelerate torchvision pillow bitsandbytes
    # If a 'Qwen3_5'/'Qwen3_6' model type is unknown, transformers is too old:
    # pip install -U "transformers @ git+https://github.com/huggingface/transformers.git@main"

Usage:
    python qwen_vision_test.py 9b  /path/to/image.jpg "Describe this image."
    python qwen_vision_test.py 4b  /path/to/image.jpg "What is in this image?"
    python qwen_vision_test.py 35b /path/to/image.jpg "Read any text in this image."

    # Run all three on the same image to compare quality + speed:
    python qwen_vision_test.py all /path/to/image.jpg "Describe this image."
"""

import sys
import time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---------------------------------------------------------------------------
# 1. EDIT THESE PATHS to point at your local checkpoint folders.
# ---------------------------------------------------------------------------
MODELS = {
    "4b": {
        "path": "/path/to/Qwen3.5-4B",
        "mode": "bf16",          # ~8 GB. Fits a 12 GB 3060 in bf16.
    },
    "9b": {
        "path": "/path/to/Qwen3.5-9B",
        "mode": "4bit",          # ~5-6 GB at 4-bit. Best fit + fastest on a 3060.
    },
    "35b": {
        "path": "/path/to/Qwen3.6-35B-A3B",
        "mode": "4bit_offload",  # ~20 GB. WON'T fit a 12 GB 3060 -> CPU offload = slow.
    },
}

# Image token budget. Lower MAX_PIXELS if you hit OOM (fewer visual tokens).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_NEW_TOKENS = 512


def load_model(cfg):
    path, mode = cfg["path"], cfg["mode"]

    if mode == "bf16":
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, device_map="auto",
        )

    elif mode == "4bit":
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            path, quantization_config=quant, device_map="auto",
        )

    elif mode == "4bit_offload":
        # 35B on a 12 GB card: keep what fits on GPU, push the rest to CPU RAM.
        # This RUNS but is slow. For real use of the 35B on a 3060, prefer Ollama
        # (llama.cpp) which handles hybrid CPU/GPU offload far more efficiently.
        from transformers import BitsAndBytesConfig
        print("  [warn] 35B won't fit a 12 GB 3060 in VRAM. Offloading to CPU - "
              "expect SLOW generation. Ollama is the better route for this model.")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            quantization_config=quant,
            device_map="auto",
            max_memory={0: "10GiB", "cpu": "48GiB"},  # tune to your RAM
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.eval()
    return model


def run_one(key, image, prompt):
    cfg = MODELS[key]
    print(f"\n{'='*60}\nMODEL: {key}  ({cfg['path']})  mode={cfg['mode']}\n{'='*60}")

    t0 = time.time()
    model = load_model(cfg)
    processor = AutoProcessor.from_pretrained(
        cfg["path"], min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    print(f"  load time: {time.time() - t0:.1f}s")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    n_prompt = inputs["input_ids"].shape[1]

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_gen = time.time()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    gen_time = time.time() - t_gen

    new_tokens = generated.shape[1] - n_prompt
    tok_per_sec = new_tokens / gen_time if gen_time > 0 else 0.0

    output = processor.batch_decode(
        generated[:, n_prompt:], skip_special_tokens=True
    )[0].strip()

    print(f"\n--- output ---\n{output}")
    print(f"\n--- speed: {new_tokens} new tokens in {gen_time:.1f}s "
          f"= {tok_per_sec:.1f} tok/s ---")

    # Free VRAM before loading the next model in 'all' mode.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "9b"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "test.jpg"
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Describe this image in detail."

    image = Image.open(image_path).convert("RGB")

    keys = list(MODELS.keys()) if key == "all" else [key]
    for k in keys:
        run_one(k, image, prompt)


if __name__ == "__main__":
    main()