import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    a1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    a2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    x1 = max(box_1[1], box_2[1])
    x2 = min(box_1[3], box_2[3])
    y1 = max(box_1[0], box_2[0])
    y2 = min(box_1[2], box_2[2])
    if x2 <= x1 or y2 <= y1:
        i = 0
    else:
        i = (x2 - x1) * (y2 - y1)
    u = a1 + a2 - i
    iou = i / u
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file in preds.keys():
        pred = preds[pred_file]
        gt = gts[pred_file]
        for i in range(len(gt)):
            pred_matching_gt = False
            for j in range(len(pred)):
                if float(pred[j][-1]) > conf_thr:
                    p = [int(pred[j][1]), int(pred[j][0]), int(pred[j][3]), int(pred[j][2])]
                    iou = compute_iou(p, gt[i])
                    if iou >= iou_thr:
                        TP += 1
                        pred_matching_gt = True
                        break
            # If we did not find any pred matching the gt, increment the FN
            if not pred_matching_gt:
                FN += 1
        for j in range(len(pred)):
            if float(pred[j][-1]) > conf_thr:
                FP += 1
    FP -= TP

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

def get_pr(preds, gts, iou_thr):
    confidence_thrs = np.arange(0.85, .96, 0.005)
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds, gts, iou_thr=iou_thr, conf_thr=conf_thr)

    # Plot training set PR curves
    print(sum(tp_train), sum(fp_train), sum(fn_train))
    precision = tp_train / (tp_train + fp_train)
    recall = tp_train / (tp_train + fn_train)
    return precision, recall


confidence_thrs = np.arange(0.85, .96, 0.005)
precision1, recall1 = get_pr(preds_train, gts_train, 0.1)
precision2, recall2 = get_pr(preds_train, gts_train, 0.25)
precision3, recall3 = get_pr(preds_train, gts_train, 0.4)
plt.plot(confidence_thrs, precision1, label="precision, 0.25")
plt.plot(confidence_thrs, precision2, label="precision, 0.5")
plt.plot(confidence_thrs, precision3, label="precision, 0.75")
plt.legend()
plt.xlabel("Confidence Threshold")
plt.ylabel("Precision")
plt.figure()
plt.plot(confidence_thrs, recall1, label="recall, 0.25")
plt.plot(confidence_thrs, recall2, label="recall, 0.5")
plt.plot(confidence_thrs, recall3, label="recall, 0.75")
plt.legend()
plt.xlabel("Confidence Threshold")
plt.ylabel("Recall")
plt.figure()
plt.plot(precision1, recall1, label="PR, 0.25")
plt.plot(precision2, recall2, label="PR, 0.5")
plt.plot(precision3, recall3, label="PR, 0.75")
plt.legend()
plt.title("Train PR Curves")
plt.xlabel("Precision")
plt.ylabel("Recall")

if done_tweaking:
    print('Code for plotting test set PR curves.')
    precision1, recall1 = get_pr(preds_test, gts_test, 0.1)
    precision2, recall2 = get_pr(preds_test, gts_test, 0.25)
    precision3, recall3 = get_pr(preds_test, gts_test, 0.4)
    plt.figure()
    plt.plot(confidence_thrs, precision1, label="precision, 0.25")
    plt.plot(confidence_thrs, precision2, label="precision, 0.5")
    plt.plot(confidence_thrs, precision3, label="precision, 0.75")
    plt.legend()
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Precision")
    plt.figure()
    plt.plot(confidence_thrs, recall1, label="recall, 0.25")
    plt.plot(confidence_thrs, recall2, label="recall, 0.5")
    plt.plot(confidence_thrs, recall3, label="recall, 0.75")
    plt.legend()
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Recall")
    plt.figure()
    plt.plot(precision1, recall1, label="PR, 0.25")
    plt.plot(precision2, recall2, label="PR, 0.5")
    plt.plot(precision3, recall3, label="PR, 0.75")
    plt.legend()
    plt.title("Test PR Curves")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()