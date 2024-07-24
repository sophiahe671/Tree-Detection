from math import inf

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
    print('results', results['results'])
    print('box_predictions', results['box_precision'])
    model.config["validation"]["csv_file"] = sample_image_path
    model.config["validation"]["root_dir"] = root_dir
    results2 = model.trainer.validate(model)
    print('results2', results2)

    # precision = results['box_precision']
    # recall = results['box_recall']

    # F1 score
    #f1_score = 2.0 * precision * recall / (precision + recall)
    #print(f1_score, 'f1_score')

    # Grid search for optimal parameters
    max_f1_score = -inf
    max_f1_params = []

    # max_iou_score = -inf
    # max_iou_params = []

    # for overlap in patch_overlap_parameters:
    #     for threshold in iou_threshold_parameters:
    #         for patch in patch_size_parameters:
    #             tile = model.predict_tile(sample_image_path, return_plot=True, patch_overlap=overlap,
    #                                       iou_threshold=threshold, patch_size=patch)
    #
    #             # Evaluation
    #             results = model.evaluate(ground_truth_annotations, root_dir, iou_threshold=0.4)
    #             precision = results['box_precision']
    #             recall = results['box_recall']
    #
    #             # F1 score
    #             f1_score = 2.0 * precision * recall / (precision + recall)
    #
    #             # IoU Evaluation method
    #
    #             if f1_score > max_f1_score:
    #                 max_f1_score = f1_score
    #                 max_f1_params = [overlap, threshold, patch]
    #
    # print('max_f1_params', max_f1_params)
    # print('max_f1_score', max_f1_score)

    # optimal_prediction = model.predict_tile(sample_image_path, return_plot=True, patch_overlap=max_f1_params[0],
    #                           iou_threshold=max_f1_params[1], patch_size=max_f1_params[2])
    #
    plt.imshow(tile[:,:,::-1])
    plt.show()
