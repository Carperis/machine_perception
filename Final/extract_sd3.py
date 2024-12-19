import torch
import os
from diffusers import StableDiffusion3Pipeline
from utils import (
    attn_maps,
    feature_maps,
    cross_attn_init,
    init_pipeline,
    save_attention_maps,
    save_feature_maps,
)

cross_attn_init()

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16,
)
pipe.load_lora_weights("lora_finetune")
# pipe.load_lora_weights("lora_finetune_yh")
pipe.to("mps")
pipe = init_pipeline(pipe)
generator = torch.Generator(pipe.device).manual_seed(0)

prompts = [
    # "a woman with green glasses blue-dotted shirt red lips big eyes golden necklace black earrings."
    # "Rugged man with defined jawline, short dark brown hair, and piercing hazel eyes. Wears a weathered leather jacket, white shirt, dark denim jeans, and brown leather boots. Confident, relaxed stance, one hand pocketed, holding a small journal. Tall, ancient trees surround him, sunlight streaming through branches, casting dappled shadows on mossy ground."
    "a man with beard"
]
batch_size = 2
prompts = prompts * batch_size

# size = 128
size = 320
images = pipe(
    prompts,
    negative_prompt="",
    num_inference_steps=28,
    height=size,
    width=size,
    guidance_scale=7.0,
    generator=generator,
).images

# target_folder = "test_long_text_small_image"
# target_folder = "test_long_text_mid_image"
# target_folder = "test_short_text_small_image"
# target_folder = "test_short_text_mid_image"
# target_folder = "test_mid_text_small_image"
# target_folder = "test_mid_text_mid_image"
# target_folder = "test_finetune_sd3_yh_sm"
target_folder = "test_finetune_sd3_our_sm"
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

for batch, image in enumerate(images):
    image.save(f"{target_folder}/{batch}-sd3.png")

print("Attention Map Shape: " + str(attn_maps[1000]["transformer_blocks.0.attn"].shape))
print("Feature Map Shape: " + str(feature_maps[1000]["transformer_blocks.0.attn"].shape))

save_attention_maps(
    attn_maps, pipe.tokenizer, prompts, base_dir=f"{target_folder}/attn_maps", unconditional=True
)
save_feature_maps(
    feature_maps, pipe.tokenizer, prompts, base_dir=f"{target_folder}/feature_maps", unconditional=True
)
