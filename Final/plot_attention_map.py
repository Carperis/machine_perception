import glob
import matplotlib.pyplot as plt
import re
import math

# def closest_factors(n):
#     sqrt_n = int(math.sqrt(n))
#     for a in range(sqrt_n, 0, -1):  # Start from sqrt(n) and go downward
#         if n % a == 0:  # Check if a is a factor
#             b = n // a
#             return a, b
#     return 1, n  # This will only happen for n = 1

# target_folder = "test_long_text_small_image"
# target_folder = "test_long_text_mid_image"
# target_folder = "test_short_text_small_image"
# target_folder = "test_short_text_mid_image"
# target_folder = "test_mid_text_small_image"
target_folder = "test_mid_text_mid_image"


image_folder = "attn_maps/batch-0"
images = glob.glob(f"{target_folder}/{image_folder}/*.png")
images.sort(key=lambda x: int(x.split("/")[-1].split("-")[0]))
# rows, cols = closest_factors(len(images))
rows = 4
cols = len(images) // rows + 1 if len(images) % rows != 0 else len(images) // rows
fig, ax = plt.subplots(rows, cols, figsize=(7, 6))
ax = ax.ravel()
for i in range(rows * cols):
    if i >= len(images):
        ax[i].axis("off")
        continue
    image = images[i]
    token = re.search(r"-<?-?(.*?)-?>?.png", image.split("/")[-1]).group(1)
    ax[i].set_title(token)
    ax[i].imshow(plt.imread(image))
    ax[i].axis("off")
plt.show()

step_folder = f"{target_folder}/attn_maps"
steps = glob.glob(f"{step_folder}/*")
steps = [step for step in steps if "." in step]
steps.sort(key=lambda x: int(float(x.split("/")[-1])))
word = "man"
# rows, cols = closest_factors(len(steps))
rows = 4
cols = len(steps) // rows + 1 if len(steps) % rows != 0 else len(steps) // rows
fig, ax = plt.subplots(rows, cols, figsize=(7, 6))
ax = ax.ravel()
# fig.suptitle(f"Attention maps for token '{word}' across {len(steps)} steps")
for i in range(rows * cols):
    if i >= len(steps):
        ax[i].axis("off")
        continue
    step = steps[i]
    step_val = int(float(step.split("/")[-1]))
    images = glob.glob(f"{step}/batch-0/*.png")
    images.sort(key=lambda x: int(x.split("/")[-1].split("-")[0]))
    image = [image for image in images if word in image][0]
    # ax[i].set_title(f"Step {step_val}")
    ax[i].set_title(f"Step {i+1}")
    ax[i].imshow(plt.imread(image))
    ax[i].axis("off")
plt.show()
