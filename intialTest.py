from math import inf

import numpy as np
from PIL import Image
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


if __name__ == '__main__':
    # fix windows multithreading issue
    freeze_support()

    # parameter values to tune
    patch_overlap_parameters = [0, 0.1, 0.2]
    iou_threshold_parameters = [0.05, 0.07, 0.1]
    patch_size_parameters = [350, 400, 450]

    # run the deep forest default model
    model = main.deepforest()
    model.use_release()

    ground_truth_annotations = get_data('TreeTestAnnotations.csv')

    # initialize file paths
    sample_image_path = get_data('TreeTest.png')

    sample_image = Image.open(sample_image_path)
    # Check if the image has an alpha channel (4th channel)
    if sample_image.mode == 'RGBA':
        # Convert the image to RGB (remove alpha channel)
        sample_image = sample_image.convert('RGB')
        sample_image.save(sample_image_path)
    root_dir = '/Users/sophi/PycharmProjects/TreeCanopyStudy/.venv/Lib/site-packages/deepforest/data'
    tile = model.predict_tile(sample_image_path, return_plot=True, patch_overlap=0.1,
                              iou_threshold=0.07, patch_size=400)

    results = model.evaluate(ground_truth_annotations, root_dir, iou_threshold=0.4)
    iou_arr = np.array(results['results']['IoU'])
    iou_arr = np.average(iou_arr)
    print('average IoU is ', iou_arr)
    print('results', results['results']['IoU'])

    # Grid search for optimal parameters
    max_iou_score = -inf
    max_iou_params = []

    for overlap in patch_overlap_parameters:
        for threshold in iou_threshold_parameters:
            for patch in patch_size_parameters:
                tile = model.predict_tile(sample_image_path, return_plot=True, patch_overlap=overlap,
                                          iou_threshold=threshold, patch_size=patch)

                # Evaluation
                results = model.evaluate(ground_truth_annotations, root_dir, iou_threshold=0.4)

                # IoU score
                iou_arr = np.array(results['results']['IoU'])
                iou_score = np.average(iou_arr)

                if iou_score > max_iou_score:
                    max_iou_score = iou_score
                    max_iou_params = [overlap, threshold, patch]

    print('max_iou_params', max_iou_params)
    print('max_iou_score', max_iou_score)

    optimal_prediction = model.predict_tile(sample_image_path, return_plot=True, patch_overlap=max_iou_params[0],
                              iou_threshold=max_iou_params[1], patch_size=max_iou_params[2])

    plt.imshow(optimal_prediction[:,:,::-1])
    plt.show()
