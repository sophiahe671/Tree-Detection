import cv2
import os
from deepforest import get_data


# def remove_alpha_channel(image_path, output_path):
#     # Read the image
#     image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#
#     # Check if image has an alpha channel (i.e., it has 4 channels)
#     if image.shape[2] == 4:
#         # Remove the alpha channel
#         image = image[:, :, :3]
#         print(f"Alpha channel removed from {image_path}")
#     else:
#         print(f"No alpha channel found in {image_path}")
#
#     # Save the image without alpha channel
#     cv2.imwrite(output_path, image)
#
#
# # Example usage
# input_image_path = get_data("QLDmaskBefore.png")
# get_data("QLDmaskBefore.png")
# output_image_path = '/Users/sophi/PycharmProjects/TreeCanopyStudy/.venv/Lib/site-packages/deepforest/data/QLDmaskBeforeChannels.png'
#
# remove_alpha_channel(input_image_path, output_image_path)

import pandas as pd

# Load the current annotations CSV
annotations = pd.read_csv('/Users/sophi/PycharmProjects/TreeCanopyStudy/.venv/Lib/site-packages/deepforest/data/finalannotations.png')

# Ensure the CSV has the correct columns
required_columns = ['image_path', 'label', 'xmin', 'ymin', 'xmax', 'ymax']

# If necessary, rename columns to match the required format
if 'image_id' in annotations.columns:
    annotations.rename(columns={'image_id': 'image_path'}, inplace=True)

if 'class_label' in annotations.columns:
    annotations.rename(columns={'class_label': 'label'}, inplace=True)

# Check if all required columns are present
missing_columns = [col for col in required_columns if col not in annotations.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing from the CSV file: {', '.join(missing_columns)}")

# Save the modified CSV file
annotations.to_csv('/Users/sophi/PycharmProjects/TreeCanopyStudy/.venv/Lib/site-packages/deepforest/data/annotations_corrected.csv', index=False)