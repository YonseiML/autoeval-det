
import numpy as np

def weighted_boxes_fusion_raw(boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.0001):
    """
    Weighted Boxes Fusion (WBF) for unnormalized bounding boxes.

    Args:
        boxes_list: List of lists of unnormalized boxes from each detector.
                    Each box format: [x1, y1, x2, y2].
        scores_list: List of lists of confidence scores for each box.
        labels_list: List of lists of labels for each box.
        image_size: Tuple of image width and height (width, height).
        iou_thr: IoU threshold for box merging.
        skip_box_thr: Minimum score threshold for including a box in merging.

    Returns:
        final_boxes: List of fused boxes in original scale.
        final_scores: List of confidence scores for fused boxes.
        final_labels: List of labels for fused boxes.
    """
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def merge_boxes(weighted_boxes, scores, weights):
        weighted_boxes = np.array(weighted_boxes)
        scores = np.array(scores)
        weights = np.array(weights)
        
        final_box = np.average(weighted_boxes, axis=0, weights=scores * weights)
        final_score = np.sum(scores * weights) / np.sum(weights)
        
        return final_box, final_score

    final_boxes = []
    final_scores = []
    final_labels = []

    all_boxes = []
    all_scores = []
    all_labels = []
    for i in range(len(boxes_list)):
        all_boxes.extend(boxes_list[i])
        all_scores.extend(scores_list[i])
        all_labels.extend(labels_list[i])

    indices = [i for i, score in enumerate(all_scores) if score >= skip_box_thr]
    all_boxes = [all_boxes[i] for i in indices]
    all_scores = [all_scores[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    used_indices = set()
    for i in range(len(all_boxes)):
        if i in used_indices:
            continue

        box_group = [all_boxes[i]]
        score_group = [all_scores[i]]
        label_group = [all_labels[i]]
        used_indices.add(i)

        for j in range(i + 1, len(all_boxes)):
            if j in used_indices:
                continue

            if all_labels[i] == all_labels[j] and calculate_iou(all_boxes[i], all_boxes[j]) > iou_thr:
                box_group.append(all_boxes[j])
                score_group.append(all_scores[j])
                used_indices.add(j)

        final_box, final_score = merge_boxes(box_group, score_group, weights=[1] * len(score_group))
        final_boxes.append(final_box)
        final_scores.append(final_score)
        final_labels.append(all_labels[i])

    return final_boxes, final_scores, final_labels
