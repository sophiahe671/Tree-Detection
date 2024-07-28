from PIL import Image
from deepforest import main
from deepforest import get_data
import pandas as pd

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

def match_boxes_and_compute_iou(ground_truth_df, predictions_df):
    iou_scores = []
    for _, ground_truth_row in ground_truth_df.iterrows():
        gt_box = {
            "xmin": ground_truth_row["xmin"], 
            "xmax": ground_truth_row["xmax"], 
            "ymin": ground_truth_row["ymin"], 
            "ymax": ground_truth_row["ymax"]
        }
        max_iou = 0
        best_box = False
        for _, prediction_row in predictions_df.iterrows():
            pred_box = {
                "xmin": prediction_row["xmin"], 
                "xmax": prediction_row["xmax"], 
                "ymin": prediction_row["ymin"], 
                "ymax": prediction_row["ymax"]
            }
            iou = compute_iou(gt_box, pred_box)
            if iou >= max_iou:
                best_box = pred_box
                max_iou = iou
        iou_scores.append({
            "gt_box": gt_box,
            "pred_box": best_box,
            "iou": max_iou
        })

    # Convert to DataFrame
    iou_df = pd.DataFrame(iou_scores)

    # Sort by IoU in descending order
    # iou_df_sorted = iou_df.sort_values(by="iou", ascending=False).reset_index(drop=True)

    average_iou = iou_df["iou"].mean()
    print(f"Average IoU: {average_iou}")

    return average_iou



if __name__ == '__main__':
    model = main.deepforest()
    model.use_release()

    ground_truth_csv = get_data('@-41.8342307,147.1550574,428m_annotations.csv')

    sample_image_path = get_data('@-41.8342307,147.1550574,428m.png')
    output_image_path = get_data('@-41.8342307,147.1550574,428m_rgb.jpg')

    image = Image.open(sample_image_path).convert("RGB")
    image.save(output_image_path)

    predictions = model.predict_tile(output_image_path, return_plot=False, patch_overlap=0.1, iou_threshold=0.07, patch_size=400)

    if not predictions.empty:
        output_csv_path = "predictions.csv"
        predictions.to_csv(output_csv_path, index=False)
        print(f"Predictions saved to {output_csv_path}")
    else:
        print("No predictions to save.")

    ground_truth_df = pd.read_csv(ground_truth_csv)
    predictions_df = pd.read_csv(output_csv_path)

    # print("Ground Truth Annotations:")
    # print(ground_truth_df.head())
    # print("\nPredicted Annotations:")
    # print(predictions_df.head())

    iou_results = match_boxes_and_compute_iou(ground_truth_df, predictions_df)
    print("\nIoU Mean:")
    print(iou_results)

    # iou_results_csv = "iou_results.csv"
    # iou_results.to_csv(iou_results_csv, index=False)
    # print(f"IoU results saved to {iou_results_csv}")