import numpy as np
import pandas as pd


def preprocess_phenotype(phenotype, drug_doses, drug_names, end_plateau_threshold=0.01,
                         drug_2_starting_plateau_finish={}, normalize_by_control=False):
    """
    phenotype : pandas.Series or numpy.array
        vector with phenotype values

    drug_doses : pandas.Series or numpy.array
        vector with phenotype corresponding drug doses (length should be the same as phenotype)

    drug_names: pandas.Series or numpy.array
        vector with phenotype corresponding drug names (length should be the same as phenotype)

    end_plateau_threshold : float
        the lowest possible value of phenotype. If lower encountered, sets it equal to the minimal value
        higher than this thresholds

    drug_2_starting_plateau_finish : dict or pd.DataFrame
        dict from drug_names variable (all should be included) to the dose value at which starting plateau ends

    normalize_by_control : bool
        if True, all values of phenotype for each drug are normalized by the control value of this drug
    """
    new_phenotype = phenotype.copy()
    unique_drugs = drug_names.unique()
    for drug_name in unique_drugs:
        current_drug_mask = drug_names == drug_name
        current_drug_phenotype = phenotype[current_drug_mask]
        current_drug_doses = drug_doses[current_drug_mask]

        current_control = current_drug_phenotype[current_drug_doses == 0].item()
        current_drug_phenotype[current_drug_phenotype > current_control] = current_control

        end_plateau_dose = current_drug_doses[current_drug_phenotype < end_plateau_threshold].astype(float).min()
        if end_plateau_dose is not np.nan:
            current_drug_phenotype[current_drug_doses > end_plateau_dose] = \
                current_drug_phenotype[current_drug_doses == end_plateau_dose].item()

        start_plateau_finish_dose = drug_2_starting_plateau_finish[drug_name]

        if normalize_by_control:
            current_drug_phenotype /= current_control
            current_drug_phenotype[current_drug_doses <= start_plateau_finish_dose] = 1.
        else:
            current_drug_phenotype[current_drug_doses <= start_plateau_finish_dose] = current_control

        new_phenotype[current_drug_mask] = current_drug_phenotype

    return new_phenotype


def good_medicines_and_doses_mask(drug_doses: pd.Series, drug_names: pd.Series,
                                  medicines_with_suspicious_last_doses: list,
                                  bad_medicines: list,
                                  exclude_control=True):
    bad_last_dose_mask = (drug_doses == 10) & (drug_names.apply(lambda x: x in medicines_with_suspicious_last_doses))
    final_mask = (~bad_last_dose_mask) & drug_names.apply(lambda x: x not in bad_medicines)
    if exclude_control:
        final_mask = final_mask & (drug_doses != 0)
    return final_mask


def find_mask_of_k_largest_doses_with_no_phenotype(y_with_plateau, drug_names, drug_doses, plateau=0.7, k=1):
    unique_drugs = drug_names.unique()
    interest_indices = []
    for drug in unique_drugs:
        drug_mask = drug_names == drug
        y_drug_values = y_with_plateau[drug_mask]
        current_drug_doses = drug_doses[drug_mask]
        drug_plateau_mask = (y_drug_values == plateau).values.ravel()

        plateau_drug_doses = current_drug_doses[drug_plateau_mask]
        non_plateau_drug_doses = current_drug_doses[~drug_plateau_mask]

        drug_interest_indices = list(plateau_drug_doses.index)[:k] + list(non_plateau_drug_doses.index)
        interest_indices += drug_interest_indices
    mask = np.zeros_like(y_with_plateau, dtype=bool)
    mask[interest_indices] = True
    return mask
