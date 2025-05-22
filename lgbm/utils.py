import numpy as np
import pandas as pd


def model_performance(y_test, y_pred, n=1000):
    """
    Routine for computing performance indicators of a given model based on test values and predicted values

    Args:
        y_test (np.array): testing dataset's class
        y_pred (np.array): predicted class

    Returns:
        (tuple): Confusion matrix thresholds, false positive rate, true positive rate, purity, accuracy
    """
    thresholds = np.linspace(start=0, stop=1, num=n)  # cut limits
    tp = np.zeros(shape=thresholds.size)  # signal selected
    fn = np.zeros(shape=thresholds.size)  # signal rejected
    fp = np.zeros(shape=thresholds.size)  # background rejected
    tn = np.zeros(shape=thresholds.size)  # background selected

    y_pred1 = y_pred[y_test == 1]
    y_pred0 = y_pred[y_test == 0]

    for i, thr in enumerate(thresholds):
        tp[i] = np.where(y_pred1 >= thr)[0].size
        fn[i] = np.where(y_pred1 < thr)[0].size
        fp[i] = np.where(y_pred0 > thr)[0].size
        tn[i] = np.where(y_pred0 <= thr)[0].size

    tpr = tp / (
        tp + fn
    )  # HEP: Efficiency, IR (Information Retrieval): Recall, Medicine: Sensitivy
    fpr = 1 - (
        fp / (fp + tn)
    )  # HEP: Background Rejection, IR: -, Medicine: Specificity
    ppv = tp / (tp + fp)  # HEP: Purity, IR: Precision, Medicine: -
    acc = (tp + tn) / (tp + fn + tn + fp)  # Accuracy

    return pd.DataFrame(
        {"threshold": thresholds, "fpr": fpr, "tpr": tpr, "ppv": ppv, "acc": acc}
    )


def compute_purity_cutflow(thresholds, fpr, tpr, ppv, acc):
    """
    Selecting indicators based on purity cuts from 10% to 90%
    """
    purity_cut = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        ppv[np.nanargmax(ppv * tpr)],
    ]
    cutflow = pd.DataFrame()
    cutflow["idx"] = [np.nanargmax(ppv >= cut) for cut in purity_cut]
    cutflow["purity_cut"] = purity_cut
    cutflow["purity"] = [ppv[purity] for purity in cutflow.idx]
    cutflow["threshold"] = [thresholds[purity] for purity in cutflow.idx]
    cutflow["efficiency"] = [tpr[purity] for purity in cutflow.idx]
    cutflow["bkg_rejection"] = [fpr[purity] for purity in cutflow.idx]
    cutflow["accuracy"] = [acc[purity] for purity in cutflow.idx]
    return cutflow.drop(["idx"], axis=1)


def compute_best_thr(thresholds, tpr, ppv):
    """
    Best purity cut baseed in max(purity*efficiency)
    """
    bst_idx = np.nanargmax(ppv >= ppv[np.nanargmax(ppv * tpr)])
    return thresholds[bst_idx]
