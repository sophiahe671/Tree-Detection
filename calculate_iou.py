from PIL import Image
from deepforest import main
from deepforest import get_data
import pandas as pd

# Compute the Intersection over Union for two boxes
def compute_iou(box1, box2):

    # Find the intersection area of the two boxes (The box that lies within both boxes)
    xmin_max = max(box1['xmin'], box2['xmin'])
    ymin_max = max(box1['ymin'], box2['ymin'])
    xmax_min = min(box1['xmax'], box2['xmax'])
    ymax_min = min(box1['ymax'], box2['ymax'])

    inter_area = max(0, xmax_min - xmin_max) * max(0, ymax_min - ymin_max)

    # Find the union area of the two boxes (total area in at least one box)
    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])

    # Subtract out intersection to aviod double counting it
    union_area = box1_area + box2_area - inter_area

    # Get Intersection over Union (IoU)
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

# Find the mean IoU for the two dataframes, usine df1 as the primary set of boxes
# This function alone has poor recall, as it ignores boxes in df2 that have no good match in df1
# To make this function better, it should be called again which the parameters swtiched.
def match_boxes_and_compute_iou(df1, df2):
    # For each box in df1, find the box in df2 that gives the maximum IoU for the pair

    iou_scores = []
    # Parse through each box in DF1 and extract the coordinates
    for _, df1_dict in df1.iterrows():
        df1_box = {
            "xmin": df1_dict["xmin"], 
            "xmax": df1_dict["xmax"], 
            "ymin": df1_dict["ymin"], 
            "ymax": df1_dict["ymax"]
        }
        max_iou = 0
        best_df2_box = False
        # Parse throguh each box in df2 and extract the coordinates
        for _, df2_dict in df2.iterrows():
            df2_box = {
                "xmin": df2_dict ["xmin"], 
                "xmax": df2_dict ["xmax"], 
                "ymin": df2_dict ["ymin"], 
                "ymax": df2_dict ["ymax"]
            }
            # Save the best IoU and cooresponding df2 box found so far
            iou = compute_iou(df1_box, df2_box)
            if iou >= max_iou:
                best_df2_box = df2_box
                max_iou = iou
        iou_scores.append({
            "df1_box": df1_box,
            "df2_box": best_df2_box,
            "iou": max_iou
        })

    # Convert IoU score to DataFrame
    iou_df = pd.DataFrame(iou_scores)

    # Find average IoU over all the df1 boxes
    average_iou = iou_df["iou"].mean()
    return average_iou


if __name__ == '__main__':
    model = main.deepforest()
    model.use_release()

    # Extract the ground truth data from csv
    ground_truth_csv = get_data('@-41.8342307,147.1550574,428m_annotations.csv')

    # Get the un-annotated sample photo
    sample_image_path = get_data('@-41.8342307,147.1550574,428m.png')

    # Convert png to jpg
    output_image_path = get_data('@-41.8342307,147.1550574,428m_rgb.jpg')
    image = Image.open(sample_image_path).convert("RGB")
    image.save(output_image_path)

    predictions = model.predict_tile(output_image_path, return_plot=False, patch_overlap=0.1, iou_threshold=0.07, patch_size=400)

    # Save predictions as csv
    if not predictions.empty:
        output_csv_path = "predictions.csv"
        predictions.to_csv(output_csv_path, index=False)
        print(f"Predictions saved to {output_csv_path}")
    else:
        print("No predictions to save.")

    ground_truth_df = pd.read_csv(ground_truth_csv)
    predictions_df = pd.read_csv(output_csv_path)

    gt_pred_iou = match_boxes_and_compute_iou(ground_truth_df, predictions_df)
    print(f"Ground truth compared to Predictions IoU: {gt_pred_iou}")

    pred_gt_iou = match_boxes_and_compute_iou(predictions_df, ground_truth_df)
    print(f"Predictions compared to Ground Truth IoU: {pred_gt_iou}")

    average_iou = (gt_pred_iou + pred_gt_iou) / 2
    print("Average IoU:")
    print(average_iou)
