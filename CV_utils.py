import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


def drug_fold(drug_names):
    """
    Generator yielding indices of an array corresponding to a particular drug.

    drug_names : np.array or pd.Series with drug names (maybe with repetitions)
    """
    unique_drugs = drug_names.unique()
    for drug in unique_drugs:
        drug_mask = drug_names == drug
        train_indices = drug_names[~drug_mask].index
        test_indices = drug_names[drug_mask].index
        yield train_indices, test_indices


def CV_regressor(model, X: pd.DataFrame, y: pd.Series,
                 drug_names, metric=mean_absolute_error, sample_weight=None, **fit_kwargs):
    train_metrics = []
    test_metrics = []
    num_features = []
    fold_weights = []

    res_pred = pd.Series(np.zeros_like(y), index=y.index)
    significant_features = np.zeros(X.shape[1])

    for train_index, test_index in drug_fold(drug_names):

        X_train, y_train = X.loc[train_index].values, y.loc[train_index].values
        X_test, y_test = X.loc[test_index].values, y.loc[test_index].values

        if sample_weight is None:
            model.fit(X_train, y_train, **fit_kwargs)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight[train_index], **fit_kwargs)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        test_score = metric(y_test, y_pred_test)
        train_score = metric(y_train, y_pred_train)

        train_metrics.append(train_score)
        test_metrics.append(test_score)

        res_pred.loc[test_index] = y_pred_test

        # if model is not of linear regression type, the coef_ attribute may not exist and so some different way
        # to measure importances of each kinase should be used (e.g. feature_importance_ in Decision tree algorithms)

        if hasattr(model, "coef_"):
            significant_features += model.coef_
        elif hasattr(model, "feature_importance_"):
            significant_features += model.feature_importance_
        else:
            raise NotImplementedError("Feature importance attribute is not known for the given model")

    significant_features /= len(np.unique(drug_names))
    significant_features_dict = {feat: abs(coef) for feat, coef in zip(X.columns, significant_features)}

    pearson_cor = pearsonr(y, res_pred)[0]
    final_r2_score = r2_score(y, res_pred)

    return {'r2': round(final_r2_score, 3),
            'cor': round(pearson_cor, 3),
            'train': np.round(np.mean(train_metrics), 3),
            'test': np.round(np.mean(test_metrics), 3)}, \
        res_pred, \
        sorted(significant_features_dict.items(), key=lambda x: x[1], reverse=True)
