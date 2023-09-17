from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
import time
import json


def init_metric_temp():
    return {
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
    }


def init_paths(experiment_id, root):
    experiment_outputs_path = os.path.join(root, experiment_id)
    curves_path = os.path.join(experiment_outputs_path, "curves")
    data_splits_path = os.path.join(experiment_outputs_path, "data_splits")
    metrics_path = os.path.join(experiment_outputs_path, " metrics")
    models_path = os.path.join(experiment_outputs_path, "models")
    setting_path = os.path.join(experiment_outputs_path, "settings.json")
    return experiment_outputs_path, curves_path, data_splits_path, metrics_path, models_path, setting_path


def setup(prefix, root, settings):
    # generate experiment id
    experiment_id = prefix + "_ex_" + str(int(time.time()))
    # output paths
    experiment_outputs_path, curves_path, data_splits_path, metrics_path, models_path, setting_path = init_paths(
        experiment_id, root)
    # create corresponding directory
    paths = [experiment_outputs_path, curves_path, data_splits_path, metrics_path, models_path]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    # writing setting
    json_object = json.dumps(settings, indent=4)
    with open(setting_path, "w") as outfile:
        outfile.write(json_object)

    return experiment_id, curves_path, data_splits_path, metrics_path, models_path


def getScores(y_pred, y):
    return (accuracy_score(y, y_pred),
            precision_score(y, y_pred, average='binary'),
            recall_score(y, y_pred, average='binary'),
            f1_score(y, y_pred, average='binary'))


def print_confusion_maxtrix(y_pred, y):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print("TN", tn)
    print("FP", fp)
    print("FN", fn)
    print("TP", tp)


def load_csv(csv_path, label):
    features = pd.read_csv(csv_path)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.fillna(0)
    features['label'] = label
    return features


def train_test_split_saving(data_splits_path, run_id, X_train, X_test, y_train, y_test):
    train_path = os.path.join(data_splits_path, str(run_id) + "_train_set.csv")
    test_path = os.path.join(data_splits_path, str(run_id) + "_test_set.csv")
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)


def metric_recording(metrics, acc_train, precision_train, recall_train, f1_train, acc_test, precision_test, recall_test,
                     f1_test):
    metrics["train_accuracy"].append(acc_train)
    metrics["train_precision"].append(precision_train)
    metrics["train_recall"].append(recall_train)
    metrics["train_f1"].append(f1_train)
    metrics["test_accuracy"].append(acc_test)
    metrics["test_precision"].append(precision_test)
    metrics["test_recall"].append(recall_test)
    metrics["test_f1"].append(f1_test)


def cal_final_metrics(metrics):
    return {
        "train_accuracy": {"mean": np.mean(metrics["train_accuracy"]), "std": np.std(metrics["train_accuracy"])},
        "train_precision": {"mean": np.mean(metrics["train_precision"]), "std": np.std(metrics["train_precision"])},
        "train_recall": {"mean": np.mean(metrics["train_recall"]), "std": np.std(metrics["train_recall"])},
        "train_f1": {"mean": np.mean(metrics["train_f1"]), "std": np.std(metrics["train_f1"])},
        "test_accuracy": {"mean": np.mean(metrics["test_accuracy"]), "std": np.std(metrics["test_accuracy"])},
        "test_precision": {"mean": np.mean(metrics["test_precision"]), "std": np.std(metrics["test_precision"])},
        "test_recall": {"mean": np.mean(metrics["test_recall"]), "std": np.std(metrics["test_recall"])},
        "test_f1": {"mean": np.mean(metrics["test_f1"]), "std": np.std(metrics["test_f1"])},
    }


def save_metrics(root_path, metrics):
    summary = {}
    result_path = os.path.join(root_path, "results.json")
    for key, values in metrics.items():
        csv_path = os.path.join(root_path, key + ".csv")
        pd.DataFrame(values).to_csv(csv_path, index=False)
        summary[key] = cal_final_metrics(values)
    print(summary)
    # writing average result
    json_object = json.dumps(summary, indent=4)
    with open(result_path, "w") as outfile:
        outfile.write(json_object)


def model_running(X_train, X_test, y_train, y_test, model, k_fold):
    cv = KFold(n_splits=k_fold)
    a_train = []
    p_train = []
    r_train = []
    f1_train = []
    for train_index, test_index in cv.split(X_train):
        k_X_train = X_train[train_index]
        k_y_train = y_train[train_index]
        k_X_test = X_train[test_index]
        k_y_test = y_train[test_index]

        model = model.fit(k_X_train, k_y_train)
        k_y_train_pred = model.predict(k_X_test)
        a, p, r, f1 = getScores(k_y_train_pred, k_y_test)
        a_train.append(a)
        p_train.append(p)
        r_train.append(r)
        f1_train.append(f1)
    y_test_pred = model.predict(X_test)
    a_test, p_test, r_test, f1_test = getScores(y_test_pred, y_test)
    return np.mean(a_train), np.mean(p_train), np.mean(r_train), np.mean(f1_train), a_test, p_test, r_test, f1_test
