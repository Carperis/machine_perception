from ncut_pytorch import NCUT
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import os
import re

def get_feature_maps(base_dir, get_x=True, get_c=True):
    entries = []
    batches = []
    c_file_names = []
    for entry in os.listdir(base_dir):
        try:
            float(entry)
            if os.path.isdir(os.path.join(base_dir, entry)):
                entries.append(entry)
            for batch in os.listdir(os.path.join(base_dir, entry)):
                if batch.startswith("batch"):
                    batches.append(batch)
                if len(c_file_names) > 0:
                    continue
                path = os.path.join(base_dir, entry, batch)
                if not os.path.isdir(path):
                    continue
                for c_file in os.listdir(path):
                    if c_file.startswith("c_feat"):
                        c_file_names.append(c_file)
        except ValueError:
            pass
    entries.sort(key=float)
    batches.sort(key=lambda x: int(x.split("-")[1]))
    c_file_names.sort(key=lambda x: int(x.split("_")[2].split("-")[0]))
    x_feature_maps = {}
    c_feature_maps = {}
    for entry in entries:
        for batch in batches:
            if get_x:
                path = os.path.join(base_dir, entry, batch, "x_feat.pt")
                with open(path, "rb") as f:
                    if x_feature_maps.get(int(float(entry))) is None:
                        x_feature_maps[int(float(entry))] = {}
                    x_feature_maps[int(float(entry))][int(batch.split("-")[1])] = torch.load(f).to(torch.float32)
            if get_c:
                for c_file in c_file_names:
                    path = os.path.join(base_dir, entry, batch, c_file)
                    token = re.search(r"-<?-?(.*?)-?>?.pt", c_file).group(1)
                    with open(path, "rb") as f:
                        if c_feature_maps.get(int(float(entry))) is None:
                            c_feature_maps[int(float(entry))] = {}
                        if c_feature_maps[int(float(entry))].get(int(batch.split("-")[1])) is None:
                            c_feature_maps[int(float(entry))][int(batch.split("-")[1])] = {}
                        c_feature_maps[int(float(entry))][int(batch.split("-")[1])][token] = torch.load(f).to(torch.float32)
    return x_feature_maps, c_feature_maps

def group_x_features(x_feature_maps, batch_no):
    steps = list(x_feature_maps.keys())
    x_features = x_feature_maps[steps[0]][batch_no]
    for step in steps[1:]:
        x_features = torch.cat((x_features, x_feature_maps[step][batch_no]), dim=0)
    return x_features # [num_steps * x_height * x_width, x_dim]

def group_c_features(c_feature_maps, batch_no):
    steps = list(c_feature_maps.keys())
    tokens = list(c_feature_maps[steps[0]][batch_no].keys())
    sentences = None  # [num_steps, num_tokens, c_dim]
    for step in steps:
        sentence = None  # [num_tokens, c_dim]
        for token in tokens:
            c_feat = c_feature_maps[step][batch_no][token].unsqueeze(0)
            if sentence is None:
                sentence = c_feat
            else:
                sentence = torch.cat((sentence, c_feat), dim=0)
        sentence = sentence.unsqueeze(0)
        if sentences is None:
            sentences = sentence
        else:
            sentences = torch.cat((sentences, sentence), dim=0)
    c_features = sentences.reshape(len(steps) * len(tokens), sentences.shape[2])
    return c_features # [num_steps * num_tokens, c_dim]

def ncut_features(features, num_eig):
    assert features.ndim == 2, "features must be n x d"
    model = NCUT(num_eig=num_eig)
    eigenvectors, eigenvalues = model.fit_transform(features)
    return eigenvectors, eigenvalues

def plot_3d(feat_3d, rgb, title, num_nodes, show=True):
    rand_indices = np.random.choice(num_nodes, 10000)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"})

    for i, ax in enumerate(axs):
        ax.scatter(*feat_3d[rand_indices].T, c=rgb[rand_indices], s=1)
        ax.view_init(elev=30, azim=45 * i)
        ax.set_title(f"View {i}")
        
    plt.suptitle(title)
    if show: plt.show()
    return fig

def plot_images(x_features_rgb, title, show=True):
    num_images = x_features_rgb.shape[0]
    subplot_width = 7
    subplot_height = num_images // subplot_width + (num_images % subplot_width != 0)
    fig, axs = plt.subplots(subplot_height, subplot_width)
    
    for i in range(num_images):
        ax = axs[i // subplot_width, i % subplot_width]
        ax.imshow(x_features_rgb[i])
        ax.axis("off")
        ax.set_title(f"Step {i+1}")
        
    plt.suptitle(title)
    if show: plt.show()
    return fig


def plot_texts(c_features_rgb, tokens, title, show=True):
    num_sentences = c_features_rgb.shape[0]
    fig, ax = plt.subplots()
    colors = [[mcolors.rgb2hex(rgb) for rgb in row] for row in c_features_rgb]

    y_pos = 1
    x_pos = 0.0
    line_height = 0.07  # Vertical space for each line
    max_width = 3  # Maximum width before wrapping to next line
    for i in range(num_sentences):
        txt = ax.text(x_pos, y_pos, f"Step {i+1}: ", color="black", fontsize=12)
        txt_width = txt.get_window_extent().width / (fig.dpi * fig.get_size_inches()[0])
        y_pos -= line_height

        for word, color in zip(tokens, colors[i]):
            text_color = "black" if sum(mcolors.hex2color(color)) > 1.5 else "white"
            txt = ax.text(
                x_pos,
                y_pos,
                word,
                color=text_color,
                bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=2),
            )
            txt_width = txt.get_window_extent().width / (
                fig.dpi * fig.get_size_inches()[0]
            )  # Calculate the width of the text in inches
            x_pos += txt_width * 1.3 + 0.015
            if x_pos > max_width:
                y_pos -= line_height
                x_pos = 0.0

        y_pos -= line_height + 0.02
        x_pos = 0.0

    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.suptitle(title)
    if show: plt.show()
    return fig

if __name__ == "__main__":
    base_dir = "/Users/sam/Desktop/Codes/machine_perception/Final/feature_maps"
    x_feature_maps, c_feature_maps = get_feature_maps(base_dir)
