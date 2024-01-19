'''
thuyhoang
'''

import os
import dlib
from scripts.align_all_parallel import align_face
import sys
from PIL import Image

from tqdm import tqdm

def run_alignment(input_folder, output_folder):
    # Load the face landmarks predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of subfolders
    subfolders = [subfolder for subfolder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, subfolder))]

    # Initialize tqdm for the main loop
    with tqdm(total=len(subfolders), desc="Processing Subfolders") as main_progress_bar:
        # Loop through all subfolders in the input folder
        for subfolder in subfolders:
            subfolder_path = os.path.join(input_folder, subfolder)

            # Get the list of image files in the subfolder
            image_files = [filename for filename in os.listdir(subfolder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not os.path.isdir(os.path.join(output_folder, subfolder)):
                os.mkdir(os.path.join(output_folder, subfolder))

            # Initialize tqdm for the nested loop
            with tqdm(total=len(image_files), desc=f"Processing {subfolder}", leave=False) as nested_progress_bar:
                # Loop through all image files in the subfolder
                for filename in image_files:
                    # Construct the full path to the image
                    image_path = os.path.join(subfolder_path, filename)

                    # Align each face in the image
                    aligned_image = align_face(filepath=image_path, predictor=predictor)
                    if isinstance(aligned_image, Image.Image):
                        # Save the aligned image to the output folder
                        output_path = os.path.join(output_folder, subfolder, filename)
                        aligned_image.save(output_path)

                        # Update the nested progress bar
                        nested_progress_bar.update(1)
                    else:
                        print(f"Can not detect landmark {filename}")

            # Update the main progress bar
            main_progress_bar.update(1)

# def run_alignment(input_folder, output_folder):
#     # Load the face landmarks predictor
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Loop through all subfolders in the input folder
#     for subfolder in os.listdir(input_folder):
#         subfolder_path = os.path.join(input_folder, subfolder)
#
#         # Check if the item in the folder is a subfolder
#         if os.path.isdir(subfolder_path):
#             if not os.path.isdir(os.path.join(output_folder, subfolder)):
#                 os.mkdir(os.path.join(output_folder, subfolder))
#             # Loop through all files in the subfolder
#             for filename in os.listdir(subfolder_path):
#                 if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     # Construct the full path to the image
#                     image_path = os.path.join(subfolder_path, filename)
#
#                     # Align each face in the image
#                     aligned_image = align_face(filepath=image_path, predictor=predictor)
#
#                     if isinstance(aligned_image, Image.Image):
#                         # Print information about the aligned image
#                         print(f"Aligned image {filename} from {subfolder} has shape: {aligned_image.size}")
#
#                         # Save the aligned image to the output folder
#                         output_path = os.path.join(output_folder, subfolder, filename)
#                         # os.path.join(output_folder, f"{filename}")
#                         aligned_image.save(output_path)
#                     else:
#                         print(f"Can not detect landmark {filename}")


if __name__ == "__main__":

    import joblib

    if len(sys.argv) < 3:
        print(
            f"usage: python {sys.argv[0]} feret_dataset_path output_align_path")
        exit()

    input_folder = sys.argv[1]
    output_folder =sys.argv[2]
    run_alignment(input_folder, output_folder)

