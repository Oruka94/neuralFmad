'''
thuyhoang
'''

import os
from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import re
import matplotlib.pyplot as plt

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

# download test image
save_path = "./fereta_aligned"



def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_testimages(save_path, viz=False):
    import os
    import matplotlib.pyplot as plt
    from PIL import Image

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

class FERETImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get the list of subdirectories
        self.classes = []  # [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.image_paths = []
        for img_name in os.listdir(root_dir):
            if img_name.endswith(".jpg") or img_name.endswith(".png"):
                label = img_name[:5]
                self.classes.append(label)  # int(d))
                self.image_paths.append(os.path.join(root_dir, img_name))

        print(f"count:{len(self.classes)}, {len(self.image_paths)}")
        print(f"pattern:{self.image_paths[0]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # @TODO


        if self.transform:
            image = self.transform(image)

        # Assume label is based on the directory name (e.g., class1 -> 0, class2 -> 1, etc.)
        # label = self.classes.index(self.selected_dir)
        label = self.classes[idx]

        return image, label


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
    print(f"ckpt.keys: {ckpt.keys()}")
    opts = ckpt['opts']
    print(f"opts:{opts}")
    print(f"latent_avg:{ckpt['latent_avg']}")
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


## pSp core
###############################################################
def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)

            print(f"W.size():{latent_to_inject.size()}")
            print(f"W:{latent_to_inject}")

            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch

#
# image to W to image
#

def calculate_psnr(img1, img2, peak=255.0):
    # assume range of img is (0,255),  (-1, +1)
    from math import log10

    x1, x2 = np.array(img1), np.array(img2)
    # print(f"peak: {x1.max()}, {x1.min()}")
    mse = np.square(x1 - x2).mean()
    return 10 * log10(peak ** 2 / mse)


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


def run_morphing_batch(net, inputs, labels, list_id=[], alpha=0.5):

    # global leanr_in_w, latent_avg
    result_batch = []
    for idx, (image_idx1, image_idx2) in enumerate(list_id):
        first_image = inputs[np.argmax(labels == image_idx1)]
        second_image = inputs[np.argmax(labels == image_idx2)]
        # print(f"morphing with {image_idx1} and {image_idx2}")
        face_morphed = morphing_face(net, first_image, second_image, alpha)
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


def run_demorphing_batch(net, lives, morpheds, alpha=0.5):
    result_batch = []
    for live, morphed in zip(lives, morpheds):
        face_demorphed = demorphing_face(net, live, morphed, alpha)
        result_batch.append(face_demorphed)

    result_batch = torch.cat(result_batch, dim=0)  # why here?
    # print(f"result_batch.shape:{result_batch.shape}")
    return result_batch

def demo_face_morphing(images_fa, labels_fa, list_id, save=False, morphed_path='morphed_fa'):
    # 0. load network
    experiment_type = 'ffhq_encode'
    net = load_network(experiment_type)
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
    img_transforms = EXPERIMENT_ARGS['transform']

    # 1. prepare input data
    images, labels = images_fa, labels_fa
    labels = np.array(labels)
    transformed_images = [img_transforms(image) for image in images]
    batched_images = torch.stack(transformed_images, dim=0)
    print(f"images:{len(images)}")


    # 2. run the pSp
    with torch.no_grad():
        tic = time.time()
        # latent_mask = []
        # result_images = run_on_batch(batched_images, net, latent_mask=latent_mask)
        result_images = run_morphing_batch(net, batched_images, labels, list_id)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    # 3. result
    if save:
        if not os.path.exists(morphed_path):
            os.makedirs(morphed_path)

        # for original_image, result_image in zip(images, result_images):
        for idx, (i,j) in enumerate(list_id):
            result_image = tensor2im(result_images[idx])

            if save:
                import cv2
                # Check if the folder exists, if not, create it
                sub_folder = os.path.join(morphed_path, f'{i}_{j}')
                if not os.path.exists(sub_folder):
                    os.makedirs(sub_folder)

                cv2.imwrite(os.path.join(sub_folder, f'{i}_{j}.png'),
                            np.array(result_image.resize((256 * 4, 256 * 4)))[:, :, ::-1])

        print(f"Save morphed images to {morphed_path}")

def demo_face_demorphing(images_fb, labels_fb, list_id, save=True, morphed_path="morphed_folder", demorphed_path="demorphed_folder"):
    # ./inversion_images : live images
    # ./morphed_faces    : morphed images

    # 0. load network
    experiment_type = 'ffhq_encode'
    net = load_network(experiment_type)
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

    # 1. prepare input data
    img_transforms = EXPERIMENT_ARGS['transform']

    # lives
    live_images, live_labels = images_fb, labels_fb
    transformed_images = [img_transforms(image) for image in live_images]
    batch_live_images = torch.stack(transformed_images, dim=0)

    # morphed
    morphed_images, morphed_labels = load_testimages(morphed_path, viz=False)
    # print(f"morphed:{len(morphed_images)}")
    label_morphed_dict = dict(zip(morphed_labels, morphed_images))
    morphed_images = [label_morphed_dict[label] for label in live_labels]
    transformed_images = [img_transforms(image) for image in morphed_images]
    batch_morphed_images = torch.stack(transformed_images, dim=0)

    # 2. run the pSp
    with torch.no_grad():
        tic = time.time()
        demorphed_images = run_demorphing_batch(net, batch_live_images, batch_morphed_images)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    # 3. result
    n = len(live_images)
    other_labels = []

    for idx in live_labels:
        for (idx1, idx2) in list_id:
            if idx1 == idx:
                other_labels.append(idx2)

    if save:
        if not os.path.exists(demorphed_path):
            os.makedirs(demorphed_path)

        for i in range(n):
            demorphed_image = tensor2im(demorphed_images[i])

            if save:
                import cv2
                sub_folder = os.path.join(demorphed_path, f"{live_labels[i]}_{other_labels[i]}")
                if not os.path.exists(sub_folder):
                    os.makedirs(sub_folder)
                cv2.imwrite(os.path.join(sub_folder, f"{live_labels[i]}_{other_labels[i]}.png"),
                            np.array(demorphed_image.resize((256 * 4, 256 * 4)))[:, :, ::-1])

        print(f"Save demorphed images to {demorphed_path}")

def find_max_values_indices(matrix):

    n = matrix.shape[0]
    max_indices = []

    for i in range(n):
        max_value = None
        max_index = None

        for j in range(n):
            if i != j:
                if max_value is None or matrix[i, j] > max_value:
                    max_value = matrix[i, j]
                    max_index = j

        max_indices.append((i, max_index))
    return np.array(max_indices)

def get_list_match(arcface_fa, labels_fa):
    cos_similarity = np.dot(arcface_fa, arcface_fa.T)
    max_indices = find_max_values_indices(cos_similarity)

    match_list = []
    for (i, j) in max_indices:
        match_list.append((labels_fa[i], labels_fa[j]))
    return match_list


if __name__ == "__main__":
    fereta_folder = "./data/fereta_aligned"
    feretb_folder = "./data/feretb_aligned"
    morphed_folder = "./data/morphed_fa"
    demorphed_folder = "./data/demorphed_fb"

    import joblib

    statistic_embedding_path = "./statistic_embedding.pkl"
    if os.path.exists(statistic_embedding_path):
        print(f"Load embedding from {statistic_embedding_path}")
        list_id, labels_fa, arcface_fa, arcface_fb, arcface_morph, arcface_demorph = joblib.load("./statistic_embedding.pkl")
    else:

        # 1. prepare input data
        images_fa, labels_fa = load_testimages(fereta_folder, viz=False)
        images_fb, labels_fb = load_testimages(feretb_folder, viz=False)

        from arcface.arcface_embedding import cal_embedding
        arcface_fa = cal_embedding(images_fa, input_size=[112, 112])

        print(f"Get list morphing")
        list_id = get_list_match(arcface_fa, labels_fa)

        if not os.path.exists(morphed_folder):
            #morph with list
            demo_face_morphing(images_fa, labels_fa, list_id=list_id, save=True, morphed_path=morphed_folder)
        if not os.path.exists(demorphed_folder):
            # demorphing with demopred image and live one
            demo_face_demorphing(images_fb, labels_fb, list_id=list_id, save=True, morphed_path=morphed_folder, demorphed_path=demorphed_folder)

        # Cal cos_similarity
        print(f"Calculate cos_similarity")
        images_morph, labels_morph = load_testimages(morphed_folder, viz=False)
        arcface_morph = cal_embedding(images_morph, input_size=[112, 112])
        label_data_dict = dict(zip(labels_morph, arcface_morph))
        arcface_morph = [label_data_dict[label] for label in labels_fa]
        arcface_morph = np.array(arcface_morph)

        images_demorph, labels_demorph = load_testimages(demorphed_folder, viz=False)
        arcface_demorph = cal_embedding(images_demorph, input_size=[112, 112])
        label_data_dict = dict(zip(labels_demorph, arcface_demorph))
        arcface_demorph = [label_data_dict[label] for label in labels_fa]
        arcface_demorph = np.array(arcface_demorph)

        arcface_fb = cal_embedding(images_fb, input_size=[112, 112])
        label_data_dict = dict(zip(labels_fb, arcface_fb))
        arcface_fb = [label_data_dict[label] for label in labels_fa]
        arcface_fb = np.array(arcface_fb)

        joblib.dump((list_id, labels_fa, arcface_fa, arcface_fb, arcface_morph, arcface_demorph), statistic_embedding_path)

    label_acc = []
    for idx in labels_fa:
        for (idx1, idx2) in list_id:
            if idx1 == idx:
                label_acc.append(idx2)
    label_data_dict = dict(zip(labels_fa, arcface_fa))
    arcface_acc = [label_data_dict[label] for label in label_acc]
    arcface_acc = np.array(arcface_acc)

    # cal cos_similarity
    cos_sim_b_a = np.dot(arcface_fa, arcface_fb.T)
    cos_sim_b_acc = np.dot(arcface_acc, arcface_fb.T)
    cos_sim_b_m = np.dot(arcface_morph, arcface_fb.T)
    cos_sim_b_d = np.dot(arcface_demorph, arcface_fb.T)

    n_pair = len(list_id)
    score_b_a = [cos_sim_b_a[i, i] for i in range(n_pair)]
    score_b_acc = [cos_sim_b_acc[i, i] for i in range(n_pair)]
    score_b_m = [cos_sim_b_m[i, i] for i in range(n_pair)]
    score_b_d = [cos_sim_b_d[i, i] for i in range(n_pair)]

    # Save to CSV
    import csv
    csv_file_path = "./statistic_results.csv"

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(["S1A label", "S2A label", "S1A vs S1B", "S1B vs S2A", "S1B vs M", "S1B vs D"])

        for row in range(n_pair):
            csv_writer.writerow([labels_fa[row], label_acc[row],
                                 round(score_b_a[row], 3),
                                 round(score_b_acc[row], 3),
                                 round(score_b_m[row], 3),
                                 round(score_b_d[row], 3),
                                 ])

    print(f"List match saved to {csv_file_path}")




