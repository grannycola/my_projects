import torch
from matplotlib import pyplot as plt
import numpy as np


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def compare_predictions(loaders, model):
    n = np.random.choice(len(loaders.test_dataset))

    image, gt_mask = loaders.test_dataset[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)

    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    visualize(
        image=image.transpose(1, 2, 0),
        ground_truth_mask=gt_mask,
        predicted_mask=pr_mask
    )
