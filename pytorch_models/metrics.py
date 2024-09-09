import numpy as np

import torch
from scipy.stats import norm
# from scipy.spatial.distance import dice
from sklearn.metrics import mean_squared_error, jaccard_score, confusion_matrix, f1_score, roc_auc_score


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

def divide_nan(numerator, denominator, replace_den_zeros=True):
    clean_numerator = np.nan_to_num(numerator, nan=0)
    # replace nan and 0 denominators with 1
    eps = 1e-10
    clean_denominator = np.nan_to_num(denominator, nan=1)
    # print(eps, clean_denominator.shape)
    if replace_den_zeros:
        clean_denominator = np.where((-eps < clean_denominator) & (clean_denominator < eps), 1, clean_denominator)
    division = clean_numerator / clean_denominator
    return division


def calculate_metrics(batch_tilewise_confusion_matrix, classes, conf_level=0.95):
    """
    Calculate macro- averaged metrics
    """

    # cumulative confusion matrix, aggregate all tile data
    cf = np.sum(batch_tilewise_confusion_matrix, axis=0)

    # initialize
    total_f1, total_precision, total_recall, total_jaccard = 0, 0, 0, 0
    micro_f1, micro_precision, micro_recall, micro_jaccard = 0, 0, 0, 0
    # predicted_positives = np.sum(cf, axis=1).tolist()  # tp + fp
    # actual_positives = np.sum(cf, axis=0).tolist()  # tp + fn
    micro_tp, micro_fp, micro_fn = 0, 0, 0
    actual_classes = 0
    #########################################
    n = np.sum(cf)  # total pixels/items
    p = cf / n
    pii = np.diag(p)  # TP
    pi_ = np.nansum(p, axis=1)  # TP + FP
    p_i = np.nansum(p, axis=0)  # TP + FN
    # micro averaged Precision, Recall and F1 for multilabel come out to be exactly the same.
    miP = miR = miF1 = np.sum(pii)

    F1i = divide_nan(2 * pii, (pi_ + p_i))  # ?
    # print(p.shape, pi_.shape, p_i.shape, pii.shape, F1i.shape)  # , pi_, p_i)
    # print(n, F1i.shape)
    # print(pi_ + p_i)
    a, b = 0, 0
    r = cf.shape[1]  # classes -> replace with actual classes
    #########################################
    for i in range(classes):
        # Calculate properties per class
        tp = cf[i][i]
        fp = np.sum(cf[:, i]) - tp
        fn = np.sum(cf[i, :]) - tp
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        # tn = np.sum(cf) - (tp + fp + fn)
        if tp + fp + fn > 0:  # class is valid: predicted or exists in gt
            actual_classes += 1
            c_precision = tp / (tp + fp) if (tp + fp) else 0
            c_recall = tp / (tp + fn) if (tp + fn) else 0
            c_f1 = 2 * (c_precision * c_recall) / (c_recall + c_precision) if (c_precision + c_recall) else 0
            c_recall += c_recall
            c_jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0
            total_f1 += c_f1
            total_precision += c_precision
            total_jaccard += c_jaccard
            total_recall += c_recall
            # only calculating over actual classes
            a += divide_nan(F1i[i] * (pi_[i] + p_i[i] - 2 * pii[i]), (pi_[i] + p_i[i]) ** 2) * (
                    (divide_nan(pi_[i] + p_i[i] - 2 * pii[i], pi_ + p_i)) + F1i[i] / 2)
            for j in range(classes):  # TODO: actual classes
                if j != i:  # ignore diagonal elements
                    b += divide_nan(p[i, j] * F1i[i] * F1i[j], ((pi_[i] + p_i[i]) * (pi_[j] + p_i[j])))
    #############################################################################
    assert actual_classes > 0, "ERROR. No actual classes found"
    maF1 = np.nansum(F1i) / actual_classes  # calculated from classwise F-1 scores
    maP = np.nansum(divide_nan(pii, pi_)) / actual_classes
    maR = np.nansum(divide_nan(pii, p_i)) / actual_classes
    maF2 = 2 * divide_nan(maP * maR, maP + maR)  # calculated from mean and precision
    # print("miP", miP, "miR", miR, "miF1", miF1, "maP", maP, "maF1", maF1, "maF2", maF2, "actual_classes",
    #       actual_classes, "maF1", "np.sum(F1i)", np.sum(F1i))  # , pi_, p_i)
    # print("n", n, "r", r)
    # for i in range(r):
    #     jj = np.delete(np.arange(r), i)
    #     for j in jj:
    #         b += p[i, j] * F1i[i] * F1i[j] / ((pi_[i] + p_i[i]) * (pi_[j] + p_i[j]))
    miF1_v = np.sum(pii) * (1 - np.sum(pii)) / n
    miF1_s = np.sqrt(miF1_v)
    # print(a.shape, b.shape)
    a = np.sum(a)
    maF1_v = 2 * (a + b) / (n * actual_classes ** 2)  # how to handle empty classes
    maF1_s = np.sqrt(maF1_v)
    z = norm.ppf(1 - (1 - conf_level) / 2)
    miF1_ci = miF1 + np.array([-1, 1]) * z * miF1_s
    maF1_ci = maF1 + np.array([-1, 1]) * z * maF1_s
    confidences = {
        # 'PointEst': [miF1, maF1],
        'SD': [miF1_s, maF1_s],
        'Lower': [miF1_ci[0], maF1_ci[0]],
        'Upper': [miF1_ci[1], maF1_ci[1]]
    }
    print(confidences)
    #############################################################################
    macro_precision = total_precision / actual_classes if actual_classes else 0
    macro_recall = total_recall / actual_classes if actual_classes else 0
    macro_f1 = total_f1 / actual_classes if actual_classes else 0
    macro_jaccard = total_jaccard / actual_classes if actual_classes else 0

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_recall + micro_precision) if (
            micro_precision + micro_recall) else 0
    micro_jaccard = micro_tp / (micro_tp + micro_fp + micro_fn) if (micro_tp + micro_fp + micro_fn) else 0

    return macro_precision, macro_recall, macro_f1, macro_jaccard, micro_precision, micro_recall, micro_f1, micro_jaccard, confidences

