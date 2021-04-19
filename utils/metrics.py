import numpy as np
from sklearn import metrics


def Jaccard(pred_arg, mask):
    pred_arg = np.argmax(pred_arg.cpu().data.numpy(), axis=1)
    mask = mask.cpu().data.numpy()

    y_true_f = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], order='F')
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2], order='F')

    intersection = np.float(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


def cla_evaluate(label, binary_score, pro_score):
    
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    auc = metrics.roc_auc_score(label, pro_score)
    CM = metrics.confusion_matrix(label, binary_score)
    MCC = metrics.matthews_corrcoef(label,binary_score)
    F1 = metrics.f1_score(label,binary_score)
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])
    return acc, auc, AP, sens, spec, MCC, F1