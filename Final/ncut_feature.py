from ncut_pytorch import NCUT, quantile_normalize
import matplotlib.pyplot as plt
import torch

unconditional = True
path = "/Users/sam/Desktop/Codes/machine_perception/Final/feature_maps/batch-0/x_feat.pt"
with open(path, "rb") as f:
    x_feat = torch.load(f).to(torch.float32)
print(x_feat.shape)
n = x_feat.shape[0]
nx = n - 154
nc = 154
h, w = 20, 20
model = NCUT(num_eig=20)
eigenvectors, eigenvalues = model.fit_transform(x_feat)
print(eigenvectors.shape, eigenvalues.shape)

fig, axs = plt.subplots(3, 4, figsize=(8, 6))
i_eig = 0 
for i_row in range(3):
    for i_col in range(1, 4):
        ax = axs[i_row, i_col]
        ax.imshow(
            eigenvectors[:, i_eig].reshape(h, w), cmap="coolwarm", vmin=-0.1, vmax=0.1
        )
        ax.set_title(f"eigenvalue_{i_eig} = {eigenvalues[i_eig]:.3f}")
        ax.axis("off")
        i_eig += 1
for i_row in range(3):
    ax = axs[i_row, 0]
    start, end = i_row * 3, (i_row + 1) * 3
    rgb = quantile_normalize(eigenvectors[:, start:end]).reshape(h, w, 3)
    ax.imshow(rgb)
    ax.set_title(f"eigenvectors {start}-{end-1}")
    ax.axis("off")
plt.suptitle("Top 9 eigenvectors of Ncut SD3 feature maps")
plt.tight_layout()
plt.show()
