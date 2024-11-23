import torch
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
pipe.to("mps")
pipe = init_pipeline(pipe)

prompts = [
    "a woman with green glasses blue-dotted shirt red lips big eyes golden necklace black earrings."
]
batch_size = 2
prompts = prompts * batch_size

images = pipe(
    prompts,
    negative_prompt="",
    num_inference_steps=28,
    height=320,
    width=320,
    guidance_scale=7.0,
).images

for batch, image in enumerate(images):
    image.save(f"./img/{batch}-sd3.png")

print("Attention Map Shape: " + str(attn_maps[1000]["transformer_blocks.0.attn"].shape))
print("Feature Map Shape: " + str(feature_maps[1000]["transformer_blocks.0.attn"].shape))

save_attention_maps(
    attn_maps, pipe.tokenizer, prompts, base_dir="attn_maps", unconditional=True
)
save_feature_maps(
    feature_maps, pipe.tokenizer, prompts, base_dir="feature_maps", unconditional=True
)
