from sklearn.preprocessing import StandardScaler

from experiment_utils import *
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Experiment Setting
settings = {
    "random_seed": 22,
    "over_sampling_rate": 1,
    "under_sampling_rate": 0.8,
    "k_fold_split": 5,
    "train_test_ratio": 0.2,
    "repeats": 10,
    "is_run_sampling": True,
    "pca_n_components": 16,
    "encoder_model_type": "deep",
    "batch_size": 16,
    "epochs": 100,
    "is_filtered_non_zero_investment": False
}

# Features set
features = [
    "pool_trading_time",
    "life_time",
    "bait_time",
    "uni_pairs",
    "pool_txs",
    "buy_txs",
    "sell_txs",
    "participants",
    "buyers",
    "sellers",
    "buy_not_sell_users",
    "sell_not_buy_users",
    "private_txs",
    "creator_txs",
    "partner",
    "users_txs",
    "senders",
    "receivers",
    "sell_partners",
    "buy_partners"]

# Random
np.random.seed(settings["random_seed"])

metrics = {
    "xgboost": init_metric_temp(),
    "knn": init_metric_temp(),
    "random_forest": init_metric_temp(),
    "svm": init_metric_temp(),
    "lightgbm": init_metric_temp(),
}


def run(X_train, X_test, y_train, y_test, metrics, model, isSTD=True):
    if isSTD:
        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)  # standardizing the data
        X_test = standard_scaler.transform(X_test)
    a_trn, p_trn, r_trn, f1_trn, a_tst, p_tst, r_tst, f1_tst = model_running(X_train,
                                                                             X_test,
                                                                             y_train.values,
                                                                             y_test.values,
                                                                             model,
                                                                             settings["k_fold_split"])
    metric_recording(metrics, a_trn, p_trn, r_trn, f1_trn, a_tst, p_tst, r_tst, f1_tst)


def load_data(seed, path, feature_list=None):
    all_data = pd.read_csv(path)
    all_data = shuffle(all_data, random_state=seed)
    X = all_data.drop(["token_address", "label"], axis=1)
    if feature_list:
        X = X[feature_list]
    y = all_data['label']
    return X, y


def lists_to_dict(keys, values):
    return {keys[i]: values[i] for i in range(len(keys))}


def opcodes_based_experiment():
    model_experimenting("OPCODE", "opcodes_based_dataset.csv")


def model_experimenting(prefix, path, features_list=None):
    experiment_id, curves_path, data_splits_path, metrics_path, models_path = setup(prefix, "experiment_results", settings)
    X, y = load_data(seed=100, path=path, feature_list=features_list)

    is_header = True
    for run_id in range(settings["repeats"]):
        seed = run_id + 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings["train_test_ratio"])
        train_test_split_saving(data_splits_path, run_id, X_train, X_test, y_train, y_test)

        columns = X_train.columns

        # KNN
        knn = KNeighborsClassifier()
        run(X_train, X_test, y_train, y_test, metrics["knn"], knn)

        # Random Forest
        random_forest = RandomForestClassifier(random_state=seed)
        run(X_train, X_test, y_train, y_test, metrics["random_forest"], random_forest)

        # SVM
        svm = SVC(kernel='poly', C=1, degree=3)
        run(X_train, X_test, y_train, y_test, metrics["svm"], svm)

        # XGBoost
        xgboost = xgb.XGBClassifier(use_label_encoder=False, random_state=seed)
        run(X_train, X_test, y_train, y_test, metrics["xgboost"], xgboost)
        xgboost_important_features = pd.DataFrame([lists_to_dict(columns, xgboost.feature_importances_)])
        xgboost_important_features.to_csv(os.path.join(metrics_path, "xgboost_important_features.csv"),
                                          mode='a',
                                          header=is_header,
                                          index=False)

        # LightGBM
        lightgbm = LGBMClassifier(random_state=seed)
        run(X_train, X_test, y_train, y_test, metrics["lightgbm"], lightgbm)
        lightgbm_important_features = pd.DataFrame([lists_to_dict(columns, lightgbm.feature_importances_)])
        lightgbm_important_features.to_csv(os.path.join(metrics_path, "lightgbm_important_features.csv"),
                                           mode='a',
                                           header=is_header,
                                           index=False)
        is_header = False
    print("Exporting metrics records")
    save_metrics(metrics_path, metrics)


if __name__ == "__main__":
    opcodes_based_experiment()
