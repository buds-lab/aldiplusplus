from functools import reduce
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class AldiEvaluationMetrics():
    """
    Provides various metrics for the evaluation of discord detectors
    """

    def get_roc_auc(self, df_true, df_pred):
        """
        Calculates column-wise the accuracy of two datasframes
        (same shape) with the same column names.

        Keyword arguments:
        df_true -- dataframe with the true values/labels
        df_pred -- dataframe with the predicted values/labels
        labels --
        Returns:
        df_accuracies -- dataframe with the column-wise accuracies
                            - index are the column names of the
                                input dataframe
                            - column is 'accuracy'
        """
        assert df_true.shape == df_pred.shape, (
            "the dataframes must have the same shape")

        df_roc_auc = pd.DataFrame(columns=['roc_auc'])
        # in order to avoid buildings where all `y_true` are either 0 or 1,
        # the entire site is evaluated as a whole
        try:
            df_roc_auc = roc_auc_score(
                df_true.values.ravel(), df_pred.values.ravel())
        except ValueError as v_error:
            print(f'ValueError w/ msg {v_error.message}')
            df_roc_auc = 0
        return df_roc_auc

    def get_accuracy(self, df_true, df_pred):
        """
        Calculates column-wise the accuracy of two datasframes
        (same shape) with the same column names.

        Keyword arguments:
        df_true -- dataframe with the true values/labels
        df_pred -- dataframe with the predicted values/labels

        Returns:
        df_accuracies -- dataframe with the column-wise accuracies
                            - index are the column names of the
                                input dataframe
                            - column is 'accuracy'
        """

        assert df_true.shape == df_pred.shape, (
            "the dataframes must have the same shape")

        df_accuracies = pd.DataFrame(index=df_true.columns,
                                     columns=['accuracy'])
        for entry in df_true.columns:
            single_accuracy = accuracy_score(df_true[entry], df_pred[entry])
            df_accuracies.at[entry, 'accuracy'] = single_accuracy

        return df_accuracies

    def get_heatmap(
        self,
        list_metric,
        list_sites,
        aldi_impl,
        metric='roc_auc',
        meter_type=0,
        p_value=0.01
    ):
        """
        Calculates a site-level accuracy heatmap

        Keyword arguments:
        list_metric -- list with all the performance metric values (e.g., roc_auc, accuracy)
        list_sites -- list with all sites
        aldi_impl -- string with the algorithm name
        metric -- string of chosen metric (e.g., 'roc_auc', 'accuracy')
        meter_type -- int of chosen meter
        p_value -- float of chosen p-value used for K-S test
        """

        df_all_metrics = pd.DataFrame(
            {'site_id': list_sites}).set_index('site_id')

        # `roc_auc` doesn't analyze each building individually, it stores the
        # value for the entire site
        if metric == 'roc_auc':
            df_all_metrics[aldi_impl] = [list_metric[site_id]
                                         for site_id in list_sites]
        else:
            df_all_metrics[aldi_impl] = [list_metric[site_id]
                                         [metric].mean() for site_id in list_sites]

        df_all_metrics.to_csv(
            f'data/results/{metric}_ai-{aldi_impl}_p{p_value}_m{meter_type}.csv')

        plt.title(f'{metric} of the different discord detectors', fontsize=18)
        fig = sns.heatmap(df_all_metrics, vmin=0, vmax=1,
                          cmap='YlGnBu').get_figure()
        fig.savefig(
            f'img/{metric}_heatmap_ai-{aldi_impl}_p{p_value}_m{meter_type}.png', format='PNG')
        plt.show()

    def get_heatmap_comparison(
        self,
        list_aldi_impl,
        list_sites,
        dict_meter_type,
        dict_p_value,
        metric='roc_auc',
        plot_name='baselines',
        fontsize=20
    ):
        """
        Compares the accuracy of different ALDI implementations in a heatmap.
        Dictionary arguments have their respective 'aldi_impl' as key.

        Keyword arguments:
        list_aldi_impl -- list with strings of algorithms names
        list_sites -- list with all sites common for
        dict_meter_type -- list with int of chosen meter (values)
        dict_p_value -- list with float of chosen p-value used for K-S test (values)
        """
        list_metric = []

        for aldi_impl in list_aldi_impl:
            p_value = dict_p_value[aldi_impl]
            meter_type = dict_meter_type[aldi_impl]
            list_metric.append(pd.read_csv(f'data/results/{metric}_ai-{aldi_impl}_p{p_value}_m{meter_type}.csv',
                                           index_col=0))

        df_metric = pd.concat(list_metric, axis=1)
        fig, ax = plt.subplots(figsize=(16, 16))
        sns.heatmap(df_acc[list_aldi_impl],
                    cmap='YlGnBu', vmin=0, vmax=1, ax=ax)

        if metric == 'roc_auc':
            metric_str = 'ROC-AUC'
        else:
            metric_str = metric

        ax.set_title(f"{metric_str} on Electricity meters",
                     fontsize=fontsize * 2)
        ax.set_xlabel("Discord detectors", fontsize=fontsize * 2)
        ax.set_ylabel("Site ID", fontsize=fontsize * 2)
        ax.tick_params(labelsize=fontsize)

        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=fontsize * 2)

        plt.xticks(rotation=90)
        plt.tight_layout()

        fig.savefig(f'img/{metric}_heatmap_{plot_name}.png', format='PNG')

    def get_class_report(
        self,
        df_true,
        df_pred,
        aldi_impl,
        level_name,
        meter_type=0,
        figsize=(10, 10),
        fontsize=40,
        path=''
    ):
        """
        Calculates the classification report and matrix based on two
        dataframes

        Keyword arguments:
        df_true -- dataframe with the true values/labels
        df_pred -- dataframe with the predicted values/labels
        aldi_impl -- string with the algorithm name
        level_name -- string with the level of comparison (e.g., all, site_id, building_id)
        meter_type -- int of chosen meter
        path -- string with relative path

        Returns:
        cf_report -- classification report generated through scitkit-learn

        """
        vector_true = df_true.values.ravel()
        vector_pred = df_pred.values.ravel()

        cm = confusion_matrix(vector_true, vector_pred,
                              labels=np.unique(vector_true))
        cf_report = classification_report(vector_true, vector_pred)

        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    #annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                    annot[i, j] = '%.1f%%' % (p)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    #annot[i, j] = '%.1f%%\n%d' % (p, c)
                    annot[i, j] = '%.1f%%' % (p)
        cm_perc = pd.DataFrame(cm_perc, index=np.unique(
            vector_true), columns=np.unique(vector_true))
        cm_perc.index.name = 'Actual'
        cm_perc.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(cm_perc,
                    cmap="YlGnBu",
                    annot=annot,
                    vmin=0,
                    vmax=100,
                    fmt='',
                    ax=ax,
                    annot_kws={"fontsize": fontsize})

#         ax.set_title(f'Confusion matrix aldi implementation\n{aldi_impl} site {level_name}',
#                      fontsize=fontsize+4)
        ax.set_xlabel("Predicted", fontsize=fontsize)
        ax.set_ylabel("Actual", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=fontsize)
        
        if path == '':
            fig.savefig(f'img/classification_report_ai-{aldi_impl}_{level_name}_m{meter_type}.png',
                        format='PNG')
        else:
            fig.savefig(f'{path}/confusion_matrix_{aldi_impl}_{level_name}.png',
                        format='PNG')
            
        plt.clf()
        return cf_report


def accuracy_barplot(  # TODO: finish
    self,
    list_aldi_impl,
    list_sites,
    dict_meter_type,
    dict_p_value,
    plot_name='baselines',
    fontsize=20
):
    """Plot accuracies of different models"""
