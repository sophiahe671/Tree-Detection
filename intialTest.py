from math import inf
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

from DeepForest.deepforest.utilities import shapefile_to_annotations


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
    root_dir = '/Users/sophi/PycharmProjects/TreeCanopyStudy/.venv/Lib/site-packages/deepforest/data'

    # Grid search for optimal parameters
    max_f1_score = -inf
    max_params = []
    for overlap in patch_overlap_parameters:
        for threshold in iou_threshold_parameters:
            for patch in patch_size_parameters:
                tile = model.predict_tile(sample_image_path, return_plot=True, patch_overlap=overlap,
                                          iou_threshold=threshold, patch_size=patch)

                # Evaluation method
                results = model.evaluate(ground_truth_annotations, root_dir, iou_threshold=0.4)
                precision = results['box_precision']
                recall = results['box_recall']
                f1_score = 2 * precision * recall / (precision + recall)

                if f1_score > max_f1_score:
                    max_f1_score = f1_score
                    max_params = [overlap, threshold, patch]

    print('max_params', max_params)
    print('max_f1_score', max_f1_score)

    optimal_prediction = model.predict_tile(sample_image_path, return_plot=True, patch_overlap=max_params[0],
                              iou_threshold=max_params[1], patch_size=max_params[2])

    plt.imshow(optimal_prediction[:,:,::-1])
    plt.show()
