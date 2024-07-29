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
                "xmin": df2_dict ["xmin"], 
                "xmax": df2_dict ["xmax"], 
                "ymin": df2_dict ["ymin"], 
                "ymax": df2_dict ["ymax"]
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

    iou_results = (match_boxes_and_compute_iou(ground_truth_df, predictions_df) + match_boxes_and_compute_iou(predictions_df, ground_truth_df)) / 2
    print("\nIoU Mean:")
    print(iou_results)

    # iou_results_csv = "iou_results.csv"
    # iou_results.to_csv(iou_results_csv, index=False)
    # print(f"IoU results saved to {iou_results_csv}")
