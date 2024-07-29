from PIL import Image
from deepforest import main
from deepforest import get_data
import pandas as pd
import matplotlib.pyplot as plt

# Compute the Intersection over Union for two boxes
def compute_iou(box1, box2):
    xmin_max = max(box1['xmin'], box2['xmin'])
    ymin_max = max(box1['ymin'], box2['ymin'])
    xmax_min = min(box1['xmax'], box2['xmax'])
    ymax_min = min(box1['ymax'], box2['ymax'])

    inter_area = max(0, xmax_min - xmin_max) * max(0, ymax_min - ymin_max)
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def match_boxes_and_compute_iou(df1, df2):
    iou_scores = []
    for _, df1_dict in df1.iterrows():
        df1_box = {
            "xmin": df1_dict["xmin"], 
            "xmax": df1_dict["xmax"], 
            "ymin": df1_dict["ymin"], 
            "ymax": df1_dict["ymax"]
        }
        max_iou = 0
        best_df2_box = False
        for _, df2_dict in df2.iterrows():
            df2_box = {
                "xmin": df2_dict["xmin"], 
                "xmax": df2_dict["xmax"], 
                "ymin": df2_dict["ymin"], 
                "ymax": df2_dict["ymax"]
            }
            iou = compute_iou(df1_box, df2_box)
            if iou >= max_iou:
                best_df2_box = df2_box
                max_iou = iou
        iou_scores.append({
            "df1_box": df1_box,
            "df2_box": best_df2_box,
            "iou": max_iou
        })

    iou_df = pd.DataFrame(iou_scores)
    average_iou = iou_df["iou"].mean()
    return average_iou

if __name__ == '__main__':
    model = main.deepforest()
    model.use_release()

    ground_truth_csv = get_data('@-41.8342307,147.1550574,428m_annotations.csv')
    sample_image_path = get_data('@-41.8342307,147.1550574,428m.png')
    output_image_path = get_data('@-41.8342307,147.1550574,428m_rgb.jpg')
    image = Image.open(sample_image_path).convert("RGB")
    image.save(output_image_path)

    patch_overlap_parameters = [0, .01, .02]
    iou_threshold_parameters = [0.055, 0.06, 0.065]
    patch_size_parameters = [460, 470, 480]

    params_iou_list = []

    for overlap in patch_overlap_parameters:
        for threshold in iou_threshold_parameters:
            for patch_size in patch_size_parameters:
                print(f"Testing with overlap {overlap}, threshold {threshold}, and patch size {patch_size}")
                predictions = model.predict_tile(sample_image_path, return_plot=False, patch_overlap=overlap, iou_threshold=threshold, patch_size=patch_size)
                
                # Convert predictions to DataFrame
                predictions_df = pd.DataFrame(predictions, columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

                # Save predictions as csv
                output_csv_path = "predictions.csv"
                predictions_df.to_csv(output_csv_path, index=False)
                print(f"Predictions saved to {output_csv_path}")

                ground_truth_df = pd.read_csv(ground_truth_csv)
                predictions_df = pd.read_csv(output_csv_path)

                # Get the average IoU of each of the boxes in the ground truth, compared to the prediction
                gt_pred_iou = match_boxes_and_compute_iou(ground_truth_df, predictions_df)
                print(f"Ground truth compared to Predictions IoU: {gt_pred_iou}")

                # Get the average IoU of each of the boxes in the prediction, compared to the ground truth
                pred_gt_iou = match_boxes_and_compute_iou(predictions_df, ground_truth_df)
                print(f"Predictions compared to Ground Truth IoU: {pred_gt_iou}")

                # Get the complete average IOU
                average_iou = (gt_pred_iou + pred_gt_iou) / 2
                print("Average IoU:")
                print(average_iou)

                params_iou_list.append({"overlap": overlap, "threshold": threshold, "patch_size": patch_size, "iou": average_iou})

    print(f"results: {str(params_iou_list)}")

    # Find the parameters that result in the maximum IoU, and show the predicion with those parameters
    max_iou_dict = max(params_iou_list, key=lambda x: x['iou'])
    print(f"max iou params: {str(max_iou_dict)}")

    optimal_prediction = model.predict_tile(sample_image_path, return_plot=True, patch_overlap=max_iou_dict["overlap"],
                                            iou_threshold=max_iou_dict["threshold"], patch_size=max_iou_dict["patch_size"])

    plt.imshow(optimal_prediction[:,:,::-1])
    plt.show()
