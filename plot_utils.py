import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from matplotlib import cm


def print_cluster_report(model, cluster_labels, feature_names):
    cluster_2_max_weight = {}
    for cluster in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        max_weight_in_cluster = round(max(abs(model.coef_[mask])), 4)
        cluster_2_max_weight[cluster] = max_weight_in_cluster
        print(cluster)
        print("Num elems in cluster", mask.sum())
        print("Max abs weight", max_weight_in_cluster)
        print({k: abs(round(w,4)) for k, w in zip(feature_names[mask], model.coef_[mask])})
        print()
    return cluster_2_max_weight


def plot_each_col_X_vs_y(X, y, num_cols=10, figsize=(60,80)):
    assert X.shape[0] == y.shape[0]

    num_rows = int(np.ceil(X.shape[1] / num_cols))

    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    ax = ax.reshape(-1)
    fig.tight_layout(w_pad=2.0, h_pad=4.0)
    for i, kinase_name in enumerate(sorted(X.columns)):
        x = X[kinase_name]
        ax[i].scatter(x, y)
        k, b = np.polyfit(x, y, 1)
        ax[i].plot(x, k * x + b, c="b")
        ax[i].set_ylabel("phenotype")
        ax[i].set_xlabel("kinase inhibition")
        ax[i].set_title(kinase_name)
    for i in range(X.shape[1], num_cols * num_rows):
        ax[i].set_visible(False)
    plt.show()


def plot_regressed_line(y, y_pred, points_names, y_drug_names=None, lims=(0, 1),
                        if_color_points=True, if_plot_diag_line=True):
    assert len(y) == len(y_drug_names)
    plt.figure(figsize=(8, 8))
    # plt.scatter(y_reg[weight_nonzero_mask], y_reg_pred[weight_nonzero_mask] / weight[weight_nonzero_mask])
    if if_plot_diag_line:
        plt.plot(np.linspace(0, 0.7, 100), np.linspace(0, 0.7, 100), c='r')

    if if_color_points:
        unique_drugs = y_drug_names.unique()
        colors = cm.Dark2_r(np.linspace(0, 1, len(unique_drugs)))
        drug_2_colors = {drug: color for drug, color in zip(unique_drugs, colors)}

        plt.scatter(y, y_pred, color=[drug_2_colors[drug] for drug in y_drug_names])
    else:
        plt.scatter(y, y_pred)

    plt.ylabel('predict')
    plt.xlabel('true')
    plt.xlim(lims)
    plt.ylim(lims)
    crs = mplcursors.cursor(hover=True)
    #     crs.connect("add", lambda sel: sel.annotation.set_text(points_names[int(sel.target.index)]))
    crs.connect("add", lambda sel: sel.annotation.set_text(points_names[int(sel.target.index)]))
    plt.show()


def plot_medicines(ys, drug_doses, drug_names, num_cols=10, num_rows=6, log_dose=True,
                   save_name='medicines_curves.pdf', save=True):
    """
    Plots phenotype values vs dose for each drug. Several phenotypes can also be drawn on each plot for comparison.
    ys : list of phenotype vectors to plot

    drug_names : pandas.Series or numpy.array
        vector with phenotype corresponding drug names (length should be the same as phenotype)
    """
    unique_drugs = drug_names.unique()
    fig, ax = plt.subplots(num_cols, num_rows, figsize=(30, 40))
    ax = ax.reshape(-1)
    fig.tight_layout(w_pad=2.0, h_pad=4.0)

    for i, drug_name in enumerate(sorted(unique_drugs)):
        drug_mask = drug_names == drug_name
        sorted_doses = drug_doses[drug_mask].sort_values()
        if log_dose:
            sorted_doses = sorted_doses[sorted_doses != 0].apply(lambda x: np.log10(float(x)))

        for y in (ys if isinstance(ys, list) else [ys]):
            ax[i].plot(sorted_doses, y[sorted_doses.index])
            ax[i].scatter(sorted_doses, y[sorted_doses.index])

        ax[i].set_title(f'{drug_name} ({len(sorted_doses)} given doses)')
        ax[i].set_ylim((0 - 1e-1, 1 + 1e-1))
        ax[i].set_ylabel('phenotype', fontsize=18)
        if log_dose:
            ax[i].set_xlim((np.log10(0.01) - 1e-1, np.log10(10) + 1e-1))
            ax[i].set_xlabel('log-doses (uM)', fontsize=18)
        else:
            ax[i].set_xlim(0 - 1e-1, 10 + 1e-1)
            ax[i].set_xlabel('doses (uM)', fontsize=18)
    for i in range(len(unique_drugs), num_cols * num_rows):
        ax[i].set_visible(False)
    if save:
        plt.savefig(save_name, pad_inches=1, bbox_inches='tight')


def plot_one_medicine(ys, drug_doses, drug_mask, title="drug_plot", labels=None, save=False):
    """
    Identical to function plot_medicines, but working for a single drug
    """
    sorted_doses = drug_doses[drug_mask].sort_values()
    sorted_doses = sorted_doses[sorted_doses != 0].apply(lambda x: np.log10(float(x)))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for i, y in enumerate(ys if isinstance(ys, list) else [ys]):
        if labels is None:
            ax.plot(sorted_doses, y[sorted_doses.index])
            ax.scatter(sorted_doses, y[sorted_doses.index])
        else:
            ax.plot(sorted_doses, y[sorted_doses.index], label=labels[i])
            ax.scatter(sorted_doses, y[sorted_doses.index])
    #     ax.set_title(f'{drug_name} ({len(sorted_doses)} given doses)')
    ax.set_ylim((0 - 1e-1, 1 + 1e-1))
    ax.set_ylabel('phenotype', fontsize=22)
    ax.set_xlim((np.log10(0.01) - 1e-1, np.log10(10) + 1e-1))
    ax.set_xlabel('log-doses (uM)', fontsize=22)
    ax.legend(fontsize=20)
    ax.set_title(title)
    if save:
        plt.savefig(title + '.png', pad_inches=1, bbox_inches='tight', dpi=100)
