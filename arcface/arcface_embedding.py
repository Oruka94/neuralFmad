import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone import Backbone
import os
import numpy as np

from tqdm import tqdm



def cal_embedding(images, input_size=[112, 112]):
    embedding_size = 512
    model_root = os.path.join("checkpoint", "backbone_ir50_ms1m_epoch120.pth")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    # check model paths
    assert os.path.exists(model_root)

    # define image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)), #resize 256x256 to 112x112
            transforms.Resize(
                [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
    )


    # load backbone weigths from a checkpoint
    backbone = Backbone(input_size)
    backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
    backbone.to(device)
    backbone.eval()

    # get embedding for each face
    embeddings = np.zeros([len(images), embedding_size])

    with torch.no_grad():
        for id, image in enumerate(images):
            image = transform(image)
            image = image.unsqueeze(0)
            embeddings[id] = F.normalize(backbone(image.to(device))).cpu()

    return embeddings