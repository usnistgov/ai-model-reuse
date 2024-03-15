import numpy as np

import torch
from scipy.stats import norm
# from scipy.spatial.distance import dice
from sklearn.metrics import mean_squared_error, jaccard_score, confusion_matrix, f1_score, roc_auc_score


def confusionmat(predictions, masks, classes):
    # print(f"predictions: {predictions.shape}, masks: {masks.shape}")
    matrix = np.zeros((classes, classes))
    for i in range(predictions.shape[0]):  # batch
        pred = torch.argmax(predictions[i], 0)
        pred = pred.cpu().detach().numpy().astype(np.uint8)
        mask = torch.squeeze(masks[i], 0)
        mask = mask.cpu().detach().numpy().astype(np.uint8)
        pred_flat = pred.flatten()
        mask_flat = mask.flatten()
        assert pred.shape == mask.shape, f"Dimensions mismatch predicted: {pred.shape}, Mask: {mask.shape}"
        # matrix += confusion_matrix(pred.flatten(), mask.flatten(), labels=classes)
        for h in range(pred_flat.shape[0]):
            matrix[pred_flat[h]][mask_flat[h]] += 1
    return matrix


def get_accuracy_batch(predictions, masks):
    total_acc = 0
    length = predictions.shape[0]
    for i in range(predictions.shape[0]):
        pred = torch.argmax(predictions[i], 0)
        pred = pred.cpu().detach().numpy().astype(np.uint8)
        mask = torch.squeeze(masks[i], 0)
        mask = mask.cpu().detach().numpy().astype(np.uint8)
        matched = np.sum(pred == mask)
        total_acc += (matched / pred.size)
    avg_acc = total_acc / length
    return avg_acc


# def get_dice_batch(predictions, masks):
#     total_dice = 0
#     length = predictions.shape[0]
#     for i in range(predictions.shape[0]):
#         pred = torch.argmax(predictions[i], 0)
#         pred = pred.cpu().detach().numpy().astype(np.uint8)
#         mask = torch.squeeze(masks[i], 0)
#         mask = mask.cpu().detach().numpy().astype(np.uint8)
#         jaccard_coeff = jaccard_score(mask.flatten(), pred.flatten(), average='micro')
#         #  https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
#         dice_coeffs = (2 * jaccard_coeff) / (jaccard_coeff + 1)
#         # print("DICE", dice_coeffs, pred.shape, mask.shape)
#         total_dice += dice_coeffs
#     avg_dice = total_dice / length
#     return avg_dice


def calculate_metrics(batch_tilewise_confusion_matrix, classes, conf_level=0.95):
    """
    Calculate macro- averaged metrics
    """

    # cumulative confusion matrix, aggregate all tile data
    cf = np.sum(batch_tilewise_confusion_matrix, axis=0)

    # initialize
    total_f1, total_precision, recall_scores, total_jaccard = 0, 0, 0, 0
    micro_f1, micro_precision, micro_recall, micro_jaccard = 0, 0, 0, 0
    # predicted_positives = np.sum(cf, axis=1).tolist()  # tp + fp
    # actual_positives = np.sum(cf, axis=0).tolist()  # tp + fn
    micro_tp, micro_fp, micro_fn = 0, 0, 0
    actual_classes = 0
    #########################################
    n = np.sum(cf)
    p = cf / n
    pii = np.diag(cf)
    pi_ = np.sum(cf, axis=1)
    p_i = np.sum(cf, axis=0)
    # micro averaged Precision, Recall and F1 for multilabel come out to be exactly the same.
    miP = np.sum(pii)
    miR = np.sum(pii)
    miF1 = np.sum(pii)
    F1i = 2 * pii / (pi_ + p_i)  # ?
    a, b = 0, 0
    r = cf.shape[1]  # classes -> replace with actual classes
    #########################################
    for i in range(classes):
        tp = cf[i][i]
        fp = np.sum(cf[:, i]) - tp
        fn = np.sum(cf[i, :]) - tp
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        # tn = np.sum(cf) - (tp + fp + fn)
        if tp + fp + fn > 0:  # class is predicted or exists in gt
            actual_classes += 1
            c_precision = tp / (tp + fp) if (tp + fp) else 0
            c_recall = tp / (tp + fn) if (tp + fn) else 0
            c_f1 = 2 * (c_precision * c_recall) / (c_recall + c_precision) if (c_precision + c_recall) else 0
            c_recall += c_recall
            c_jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0
            total_f1 += c_f1
            total_precision += c_precision
            total_jaccard += c_jaccard
            # tp_sum += tp
            for j in range(classes):
                if j != i:  # ignore diagonal elements
                    b += p[i, j] * F1i[i] * F1i[j] / ((pi_[i] + p_i[i]) * (pi_[j] + p_i[j]))
    #############################################################################
    maF1 = np.sum(F1i) / actual_classes  # calculated from classwise F1scores
    maP = np.sum(pii / np.sum(p, axis=1)) / actual_classes
    maR = np.sum(pii / np.sum(p, axis=0)) / actual_classes
    maF2 = 2 * (maP * maR) / (maP + maR)  # calculated from mean and precision

    # for i in range(r):
    #     jj = np.delete(np.arange(r), i)
    #     for j in jj:
    #         b += p[i, j] * F1i[i] * F1i[j] / ((pi_[i] + p_i[i]) * (pi_[j] + p_i[j]))
    miF1_v = np.sum(pii) * (1 - np.sum(pii)) / n
    miF1_s = np.sqrt(miF1_v)
    maF1_v = 2 * (a + b) / (n * r ** 2)  # how to handle empty classes
    maF1_s = np.sqrt((maF1_v))

    z = norm.ppf(1 - (1 - conf_level) / 2)
    miF1_ci = miF1 + np.array([-1, 1]) * z * miF1_s
    maF1_ci = maF1 + np.array([-1, 1]) * z * maF1_s
    confidences = {
        # 'PointEst': [miF1, maF1],
        'SD': [miF1_s, maF1_s],
        'Lower': [miF1_ci[0], maF1_ci[0]],
        'Upper': [miF1_ci[1], maF1_ci[1]]
    }
    #############################################################################
    macro_precision = total_precision / actual_classes if actual_classes else 0
    macro_recall = recall_scores / actual_classes if actual_classes else 0
    macro_f1 = total_f1 / actual_classes if actual_classes else 0
    macro_jaccard = total_jaccard / actual_classes if actual_classes else 0

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0
    micro_f1 = 2 * (c_precision * c_recall) / (c_recall + c_precision) if (c_precision + c_recall) else 0
    micro_jaccard = micro_tp / (micro_tp + micro_fp + micro_fn) if (micro_tp + micro_fp + micro_fn) else 0

    return macro_precision, macro_recall, macro_f1, macro_jaccard, micro_precision, micro_recall, micro_f1, micro_jaccard, confidences


# def get_jaccard_batch(predictions, masks):
#     total_jaccard = 0
#     length = predictions.shape[0]
#     for i in range(predictions.shape[0]):
#         pred = torch.argmax(predictions[i], 0)
#         pred = pred.cpu().detach().numpy().astype(np.uint8)
#         mask = torch.squeeze(masks[i], 0)
#         mask = mask.cpu().detach().numpy().astype(np.uint8)
#         # print("maskshape", mask.shape,"predshape", pred.shape)
#         jaccard_coeff = jaccard_score(mask.flatten(), pred.flatten(), average='micro')
#         #  https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
#         total_jaccard += jaccard_coeff
#     avg_jaccard = total_jaccard / length
#     return avg_jaccard


def get_mse_batch(predictions, masks):
    total_mse = 0
    length = predictions.shape[0]
    for i in range(predictions.shape[0]):
        pred = torch.argmax(predictions[i], 0)
        pred = pred.cpu().detach().numpy().astype(np.uint8)
        mask = torch.squeeze(masks[i], 0)
        mask = mask.cpu().detach().numpy().astype(np.uint8)
        mse_coeffs = mean_squared_error(mask.flatten(), pred.flatten())
        total_mse += mse_coeffs
    avg_mse = total_mse / length
    return avg_mse
