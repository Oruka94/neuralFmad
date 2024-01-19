'''
thuyhoang
'''

import os
from argparse import Namespace
import time
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import re
import matplotlib.pyplot as plt
from arcface.arcface_embedding import cal_embedding

# CODE_DIR = 'neuralFmad'
# if False:
#     sys.path.insert(0, ".")
#     sys.path.insert(0, "..")
#     from datasets import augmentations
# else:
#     sys.path.append(".")
#     sys.path.append("..")

from utils.common import tensor2im, log_input_image
from models.psp import pSp

####################################################
# Inference Type parameter
####################################################
learn_in_w = None
latent_avg = None
start_from_latent_avg = None

MODEL_PATHS = {
    "ffhq_encode": {"id": "1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0", "name": "psp_ffhq_encode.pt"},
    "ffhq_frontalize": {"id": "1_S4THAzXb-97DbpXmanjHtXRyKxqjARv", "name": "psp_ffhq_frontalization.pt"},
    "celebs_sketch_to_face": {"id": "1lB7wk7MwtdxL-LL4Z_T76DuCfk00aSXA", "name": "psp_celebs_sketch_to_face.pt"},
    "celebs_seg_to_face": {"id": "1VpEKc6E6yG3xhYuZ0cq8D2_1CbT0Dstz", "name": "psp_celebs_seg_to_face.pt"},
    "celebs_super_resolution": {"id": "1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu", "name": "psp_celebs_super_resolution.pt"},
    "toonify": {"id": "1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz", "name": "psp_ffhq_toonify.pt"}
}


EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pretrained_models/psp_ffhq_encode.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "ffhq_frontalize": {
        "model_path": "pretrained_models/psp_ffhq_frontalization.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celebs_sketch_to_face": {
        "model_path": "pretrained_models/psp_celebs_sketch_to_face.pt",
        "image_path": "notebooks/images/input_sketch.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    },
    '''
    "celebs_seg_to_face": {
        "model_path": "pretrained_models/psp_celebs_seg_to_face.pt",
        "image_path": "notebooks/images/input_mask.png",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.ToOneHot(n_classes=19),
            transforms.ToTensor()])
    },
    "celebs_super_resolution": {
        "model_path": "pretrained_models/psp_celebs_super_resolution.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[16]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    '''
    "toonify": {
        "model_path": "pretrained_models/psp_ffhq_toonify.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_testimages(save_path, viz=False):
    # image_paths = [os.path.join(save_path, f) for f in os.listdir(save_path) if
    #                f.endswith(".jpg") or f.endswith(".png")]
    image_paths = []
    for subfolder in os.listdir(save_path):
        subfolder_path = os.path.join(save_path, subfolder)
        image_path = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if
                   f.endswith(".jpg") or f.endswith(".png")]
        image_paths.extend(image_path)
    image_paths.sort(key=natural_sort_key)  # Sort based on natural order

    images = []
    labels = []  # List to store labels

    if viz:
        n_cols = int(np.ceil(len(image_paths) / 2))
        fig = plt.figure(figsize=(20, 4))

    for idx, image_path in enumerate(image_paths):
        if viz:
            ax = fig.add_subplot(2, n_cols, idx + 1)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

        img = Image.open(image_path).convert("RGB")
        images.append(img)

        # Extract the first 5 characters of the image name as the label
        label = os.path.splitext(os.path.basename(image_path))[0][:5]
        # labels.append(int(label))
        labels.append(label)


        if viz:
            ax.imshow(img)

    if viz:
        plt.show()

    return images, labels


## load network
####################################################################
def load_network(experiment_type='ffhq_encode'):
    global learn_in_w, latent_avg, start_from_latent_avg

    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

    # already downloaded
    model_path = EXPERIMENT_ARGS['model_path']
    if os.path.getsize(model_path) < 1000000:
        raise ValueError("Pretrained model was unable to be downlaoded correctly!")

    # load network
    ckpt = torch.load(model_path, map_location='cpu')
    # print(f"ckpt.keys: {ckpt.keys()}")
    opts = ckpt['opts']
    # print(f"opts:{opts}")
    # print(f"latent_avg:{ckpt['latent_avg']}")
    learn_in_w = opts['learn_in_w']
    latent_avg = ckpt['latent_avg']
    start_from_latent_avg = opts['start_from_latent_avg']

    # update the training options
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('Model successfully loaded!')

    return net

#################################################################################################
# MORPHING
#################################################################################################
# 2 morphing implementations

# 1. first get the codes and blend them, then run the synthesis (the style ... should be used?)
def morphing_face(net, face1, face2, alpha):
    # get intermediate latent
    _, latent1 = net(face1.unsqueeze(0).to("cuda").float(), input_code=False, return_latents=True)
    _, latent2 = net(face2.unsqueeze(0).to("cuda").float(), input_code=False, return_latents=True)

    debug = False
    if debug:
        print(f"W{image_idx1}.size():{latent1.size()}")
        print(f"W{image_idx1}:{latent1}")
        print(f"W{image_idx2}.size():{latent2.size()}")
        print(f"W{image_idx2}:{latent2}")

    # blending
    latent = alpha * latent1 + (1 - alpha) * latent2

    # get output image with blended vector
    net.set_force_input_is_latent(True)
    face_res = net(latent, input_code=True)
    net.set_force_input_is_latent(False)

    return face_res


# 2. first get the code for the face1 and then inject it to the face2
def morphing_face2(net, face1, face2, alpha):
    # get intermediate latent
    _, latent1 = net(face1.unsqueeze(0).to("cuda").float(), input_code=False,
                     latent_mask=None, inject_latent=None, return_latents=True)

    if False:
        print(f"W{image_idx1}.size():{latent1.size()}")
        print(f"W{image_idx1}:{latent1}")

    # blending
    # get output image with blended vector
    latent_mask = list(range(0, 18))  # all StyleGan layers
    res = net(face2.unsqueeze(0).to("cuda").float(), input_code=False,
              latent_mask=latent_mask, inject_latent=latent1, alpha=alpha)

    return res
def run_morphing_batch(net, inputs1, inputs2, alpha=0.5):

    # global leanr_in_w, latent_avg
    result_batch = []
    for idx, image2 in enumerate(inputs2):
        face_morphed = morphing_face(net, inputs1, image2, alpha)
        # print(f"face_morphed.shape:{face_morphed.shape}")
        result_batch.append(face_morphed)

    result_batch = torch.cat(result_batch, dim=0)  # why here?
    # print(f"result_batch.shape:{result_batch.shape}")
    return result_batch
##########################################################################################
# Demorphing
##########################################################################################
# first get the codes and blend them, then run the synthesis (the style ... should be used?)
def demorphing_face(net, live, morphed, alpha=0.5):
    # get intermediate latent
    _, latent_live = net(live.unsqueeze(0).to("cuda").float(), input_code=False, return_latents=True)
    _, latent_morphed = net(morphed.unsqueeze(0).to("cuda").float(), input_code=False, return_latents=True)

    # blending
    latent = 2 * latent_morphed - alpha * latent_live

    # get output image with blended vector
    net.set_force_input_is_latent(True)
    face_res = net(latent, input_code=True)
    net.set_force_input_is_latent(False)

    return face_res

def get_accomplices_label(mal_idx, num_acc, embeddings_fa):
    cos_sim = np.dot(embeddings_fa[mal_idx], embeddings_fa.T)
    cos_sim = cos_sim.reshape(-1)
    # print(f"cos_sim: {cos_sim.shape}")
    top_idx = np.argsort(cos_sim)[-num_acc-1:-1]

    return top_idx


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            f"usage: python {sys.argv[0]} malicious_label num_accomplice alpha_list /n"
            f"example: python p2s2p_feret_v1.py 00070 5 0.2,0.25,0.3")
        exit()

    fereta_folder = "./data/fereta_aligned"
    feretb_folder = "./data/feretb_aligned"

    mal_label = sys.argv[1]
    num_acc = int(sys.argv[2])
    alpha_string = sys.argv[3]

    alpha_list = [float(x) for x in alpha_string.split(',')]
    num_alpha = len(alpha_list)

    #0. load folder images
    print("Load images")
    images_fa, ids_fa = load_testimages(fereta_folder, viz=False)
    ids_fa = np.array(ids_fa)

    #check mal image exist
    # print(f"np.where(ids_fa == mal_label): {np.where(ids_fa == mal_label)}")
    if len(np.where(ids_fa == mal_label)[0])<=0:
        print(f"{mal_label} not exist")
        exit()

    #cal arcface embedding
    arcface_fa = cal_embedding(images_fa, input_size=[112, 112])
    # sim_matrix = np.dot(arcface_fa, arcface_fa.T)
    # for i in range(10):
    #     print(f" sim_matrix: {sim_matrix[i,i]}")

    # find idx mal and acc images
    mal_idx = np.argmax(ids_fa == mal_label)
    top_idx = get_accomplices_label(mal_idx, num_acc, arcface_fa)

    top_label = []
    for i in range(num_acc):
        top_label.append(ids_fa[top_idx[i]])

    images_fb, ids_fb = load_testimages(feretb_folder, viz=False)
    #cal arcface embedding
    arcface_fb = cal_embedding([images_fb[mal_idx]], input_size=[112, 112])[0]

    #1. load network
    print(f"Load model")
    experiment_type = 'ffhq_encode'
    net = load_network(experiment_type)
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

    # 2. prepare input data for morphing
    img_transforms = EXPERIMENT_ARGS['transform']
    ids_fa = np.array(ids_fa)
    transformed_images = [img_transforms(image) for image in images_fa]
    batched_images = torch.stack(transformed_images, dim=0)

    mal_img = batched_images[mal_idx]
    live_img = img_transforms(images_fb[mal_idx])

    morphed_results = []
    demorphed_results = []
    cos_similarity = np.zeros((num_acc, 2 + 2*len(alpha_list)))

    print(f"Morphing and Demorphing")
    with torch.no_grad():
        for i in range(num_acc):
            acc_img_idx = top_idx[i]
            acc_img = batched_images[acc_img_idx]

            cos_similarity[i,0] = np.dot(arcface_fa[mal_idx].T, arcface_fb)
            cos_similarity[i, 1] = np.dot(arcface_fa[acc_img_idx].T, arcface_fb)
            morphed_rows = []
            demorphed_rows = []

            for j, alpha in enumerate(alpha_list):
                morphed_tensor = morphing_face(net, mal_img, acc_img, alpha).squeeze()
                morphed_img = tensor2im(morphed_tensor)
                morphed_rows.append(morphed_tensor)

                demorphed_tensor = demorphing_face(net, live_img, morphed_tensor, alpha).squeeze()
                demorphed_img = tensor2im(demorphed_tensor)
                demorphed_rows.append(demorphed_tensor)
                # 3. cal cos_sim
                # Convert PyTorch Tensor to PIL Image
                # pil_img = to_pil(morphed_row.cpu())  # Move to CPU before converting to PIL Image

                arcface_morphed = cal_embedding([morphed_img], input_size=[112, 112])[0]
                cos_similarity[i, 2 + 2*j] = np.dot(arcface_morphed.T, arcface_fb)

                arcface_demorphed = cal_embedding([demorphed_img], input_size=[112, 112])[0]
                cos_similarity[i, 2 + 2*j+1] = np.dot(arcface_demorphed.T, arcface_fb)

            morphed_results. append(morphed_rows)
            demorphed_results.append(demorphed_rows)
    morphed_results = np.array(morphed_results)
    demorphed_results = np.array(demorphed_results)

    #4. visualize
    n_col = 3+2*num_alpha
    n_row = num_acc
    for i in range(num_acc):
        mal_a_img = images_fa[mal_idx]
        mal_b_img = images_fb[mal_idx]
        acc_img = images_fa[top_idx[i]]

        plt.subplot(n_row, n_col, i*n_col+1)
        img = Image.fromarray(np.array(mal_a_img.resize(( 256 * 4, 256 * 4))))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Mal A ({mal_label})", fontsize=12)

        plt.subplot(n_row, n_col, i*n_col + 2)
        img = Image.fromarray(np.array(mal_b_img.resize((256 * 4, 256 * 4))))
        plt.imshow(img)
        plt.axis('off')
        plt.title(
            f"Mal B ({mal_label}), {cos_similarity[i, 0]:.3f}", fontsize=10)

        plt.subplot(n_row, n_col, i*n_col + 3)
        img = Image.fromarray(np.array(acc_img.resize((256 * 4, 256 * 4))))
        plt.imshow(img)
        plt.axis('off')
        plt.title(
            f"Acc ({top_label[i]}), {cos_similarity[i, 1]:.3f}", fontsize=10)

        for j in range(num_alpha):
            morphed_img = tensor2im(morphed_results[i, j])
            morphed_img = np.array(morphed_img.resize((256*4, 256*4)))
            plt.subplot(n_row, n_col, i*n_col + 4+ 2*j)
            plt.imshow(morphed_img)
            plt.axis('off')
            plt.title(
                f"Morph alpha={alpha_list[j]}, {cos_similarity[i, 2+2*j]:.3f}", fontsize=10)

            demorphed_img = tensor2im(demorphed_results[i, j])
            demorphed_img = np.array(demorphed_img.resize((256 * 4, 256 * 4)))
            plt.subplot(n_row, n_col, i * n_col + 4 + 2 * j+1)
            plt.imshow(demorphed_img)
            plt.axis('off')
            plt.title(
                f"Demorph alpha={alpha_list[j]}, {cos_similarity[i, 2 + 2*j+1]:.3f}", fontsize=10)

    plt.show()


