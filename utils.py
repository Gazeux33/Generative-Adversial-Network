import numpy as np
import matplotlib.pyplot as plt
import json
import os

import config


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def save_history(history, path, name_file):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name_file), "w+") as f:
        json.dump(history, f, indent=2)


def visualize_images(images, epoch=0, batch=0, save=True):
    images = images.clamp(0, 1)

    fig, ax = plt.subplots(1, 10, figsize=(10, 2))
    for i in range(10):
        img = images[i].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    if save:
        plt.savefig(f"{config.RESULT_DIR}/e_{epoch}_b_{batch}.png")


def plot_history(history):
    plt.figure()
    for key in history:
        plt.plot(history[key], label=key)
    plt.grid()
    plt.title('Training History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
