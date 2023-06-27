import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import combinations
import pandas as pd

"""
1. should boxplots change for log scale? - no
2. statistical significance calculation
3. statistical significance visualization
4. Cleaner plots
    a. Titles, names - DONE
    b. move legend outside/to right - DONE
5. Use most accurate model in each training (12x4 instead of 1200x4) for?
6. For model stability 
    -> Annotate the best model with special colors? - DONE
    -> Compare best model in each category? - DONE
7. For model speed 
    -> Annotate the best models with special colors? 
    -> Compare best model in each category?

48 best epochs for 48 models.
For 4 set<harmonics> = 12 values of 1. Losses, 2. 

"""


def get_best_model_df(df, metric=None, groupby=None):
    optimal_product = None

    if groupby is None:
        groupby = ['channel', 'lr', 'pretrained']
    if metric is None:
        metric = "crossentropy"
    if metric in ["CE", "crossentropy"]:
        df['optimal_metric'] = np.sqrt(df['Train_loss'] ** 2 + df['Test_loss'] ** 2)
        optimal_product = df.groupby(groupby)['optimal_metric'].idxmin()  # .reset_index()
    elif metric == "Dice":
        df['optimal_metric'] = df["Dice"]
        optimal_product = df.groupby(groupby)['optimal_metric'].idxmax()  # .reset_index()
    # print(f"minmetric { df}")
    # print(f"minmetric { df['optimal_metric']}")
    # best_id_ch = df['optimal_metric'].idxmin()
    best_df_ch = df.loc[optimal_product]
    # minloss = df.min()
    return best_df_ch


def do_stat_annotations(df, ax, pair_param, x=None, y=None, order=None):
    try:
        pairs = None
        if pair_param == "channel":
            pair_param_uniques = list(df.channel.unique())
        if pair_param == "pretrained":
            pair_param_uniques = list(df.pretrained.unique())
        if pair_param == "lr":
            pair_param_uniques = list(df.lr.unique())
        # pairs = list(set([(a, b) for a in pair_param_uniques for b in pair_param_uniques if a != b]))
        all_pairs = list(combinations(pair_param_uniques, 2))
        pairs = list(set(tuple(sorted(all_pair)) for all_pair in all_pairs))
        print("ANNO\n", df[df.isna().any(axis=1)], "\n", df.dtypes)
        annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside',line_height=0.01, text_offset=1)
        # annotator.configure(test='t-test_ind', text_format='star', loc='outside', line_height=0.01, text_offset=1)
        # annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        annotator.apply_and_annotate()
        # annotator.apply_test(nan_policy='propagate')
        # annotator.annotate()
    except Exception as e:
        print("ERROR", e)
    return ax


def plot_metrics_comparison(df, savepath, comparemetrics=None, tr_thresh=100, tst_thresh=100, illustrate_thresh=True):
    if comparemetrics is None:
        comparemetrics = ['Train_loss', 'Test_loss']
    fig = plt.figure(1)
    metric1, metric2 = comparemetrics
    # tr_thresh = 100
    # tst_thresh = 100
    df = df[df['Train_loss'] <= tr_thresh]
    df = df[df['Test_loss'] <= tst_thresh]
    if metric1 in ['Train_loss', 'Test_loss']:
        if metric2 in ['Train_loss', 'Test_loss']:
            if illustrate_thresh == True:
                x = [1, 2, 3, 4, 5]
                y = [1, 2, 3, 4, 5]
                plt.vlines(x, 0, y, linestyle="solid")
                plt.hlines(y, 0, x, linestyle="solid")
    sns.scatterplot(df, x=metric1, y=metric2, s=5, alpha=0.95, hue=df['channel'],
                    palette='Spectral') # label=f'{metric1}_vs_{metric2}'
    # plt.scatter(dataF['Train_loss'][0:], dataF['Test_loss'][0:], c='blue', s=5, label='loss comparison', hue=dataF['epoch'])
    plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    title_test = f"{metric1} vs {metric2}"
    plt.title(title_test)
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    outGraph_test = savepath + os.path.sep + f'{metric1}_vs_{metric2}_thresh{tr_thresh}-{tst_thresh}_comparison.png'
    plt.savefig(outGraph_test, bbox_inches='tight')
    plt.clf()
    plt.close()


def get_stability(df, use_best=False, metric="CE"):
    """
    Stability of a model is calculated  based on change of train and test loss over the entire model.

    :param df: dataframe containing all data
    :param use_best: present stability for the best models

    """
    channels = list(df.channel.unique())
    lrs = list(df.lr.unique())
    pts = list(df.pretrained.unique())
    residual_arr_df = pd.DataFrame({'channel': pd.Series(dtype='str'),
                                    'lr': pd.Series(dtype='float64'),
                                    'pretrained': pd.Series(dtype='bool'),
                                    'm': pd.Series(dtype='float64'),
                                    'c': pd.Series(dtype='float64'),
                                    'residual': pd.Series(dtype='float64')})
    # print("INDEX ", residual_arr_df.index, residual_arr_df.columns)
    for channel in channels:
        df_ch = df[df['channel'] == channel]
        for lr in lrs:
            df_ch_lr = df_ch[df_ch['lr'] == lr]
            for pt in pts:
                df_ch_lr_pt = df_ch_lr[df_ch_lr['pretrained'] == pt]
                # find slope of training and test curves
                mintestloss = df_ch_lr_pt['Test_loss'].min()
                mintrainloss = df_ch_lr_pt['Train_loss'].min()
                # df_testvals= df_ch_lr_pt[['epoch','Test_loss']].to_numpy()
                df_trainvals, df_testvals = df_ch_lr_pt['Train_loss'].to_numpy(), df_ch_lr_pt['Test_loss'].to_numpy()
                df_testvals = df_testvals - mintestloss
                df_trainvals = df_trainvals - mintrainloss
                A = np.vstack([df_trainvals, np.ones(len(df_trainvals))]).T
                [m, c], [tt_residual] = np.linalg.lstsq(A, df_testvals, rcond=None)[:2]
                residual_arr_df.loc[len(residual_arr_df), residual_arr_df.columns] = channel, lr, pt, m, c, tt_residual
                # r2_tt = 1 - tt_residual / (df_testvals.size * df_testvals.var())
                # print(m, c, tt_residual)
    # print(f"residual array size: {residual_arr_df}")
    ######################################################################################
    qvals = [channels, lrs, pts]
    quants = ['channel', 'lr', 'pretrained']
    i = 0

    props = {
        'boxprops': {'facecolor': None, 'edgecolor': 'black'},
        # 'boxprops': {'edgecolor': 'Spectral'},
        'medianprops': {'color': 'green'},
        'whiskerprops': {'color': 'green'},
        'capprops': {'color': 'green'}
    }
    # props={}

    ####################################################################################################################
    ## Plotting residuals for all 48 models. Comparison between a secondary parameter included.
    ####################################################################################################################
    for quant in quants:
        for hue in quants:
            if hue != quant:
                fig = plt.figure(i)
                # print(f"residual array: {residual_arr_df}")
                plt.yscale('log')
                sns.boxplot(data=residual_arr_df, x=quant, y='residual', palette="Spectral", hue=hue,
                            showfliers=False, **props)
                plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
                sns.stripplot(data=residual_arr_df, x=quant, y='residual', color="black", dodge=5, hue=hue, size=3,
                              linewidth=0, legend=False)  # legend = False for newer versions
                # , alpha = 0.5
                plt.xticks(range(len(qvals[quants.index(quant)])), labels=qvals[quants.index(quant)])
                title_test = f"Residual wrt {quant}"
                # plt.title(title_test)
                plt.xlabel(quant)
                plt.ylabel('Residual')
                outGraph_test = savepath + os.path.sep + f'Model_stability_{quant}_compare{hue}.png'
                plt.savefig(outGraph_test, bbox_inches='tight')
                plt.clf()
                plt.close()
                i += 1
    ####################################################################################################################
    ## Plotting residuals for all 48 models. Highlighted best model within each category.
    ####################################################################################################################
    props = {
        # 'boxprops': {'facecolor': 'none', 'edgecolor': 'red'},
        'boxprops': {'edgecolor': 'black'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }

    for quant in quants:
        fig = plt.figure(i)
        plt.yscale('log')
        sns.boxplot(residual_arr_df, x=quant, y="residual", palette="Spectral", showfliers=False, **props)
        # plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
        sns.stripplot(residual_arr_df, x=quant, y="residual", color="black", dodge=5, size=3,
                      linewidth=0, legend=False)
        residual_arr_df_na = residual_arr_df.dropna()
        do_stat_annotations(df=residual_arr_df_na, ax=plt.gca(), pair_param=quant, x=quant, y="residual", order=None)

        # print("HIHI")
        if use_best:
            best_df_quant = get_best_model_df(df, groupby=[quant], metric=metric)
            columns_to_match = quants
            residual_arr_best = pd.merge(residual_arr_df, best_df_quant, on=columns_to_match, suffixes=('', '_best'))
            # print(residual_arr_best)
            # print(residual_arr_df.index[best_df_quant.index])
            sns.stripplot(residual_arr_best, x=quant, y="residual", color="red", dodge=5, size=5, linewidth=0,
                          legend=False)
            for id, row in residual_arr_best.iterrows():
                # print(f"{row['channel']},\n{row['lr']}\n{row['pretrained']}", (row[quant], (row['optimal_metric'])))
                # Show optimal metric instead of residual
                plt.gca().annotate(f"{row['optimal_metric']:.4e}\n{row['channel']},\n{row['lr']}\n{row['pretrained']}",
                                   (str(row[quant]), (row['residual'])), xytext=(20, -5), textcoords='offset points',
                                   family='sans-serif', fontsize=8, color='darkslategrey')

        plt.xticks(range(len(qvals[quants.index(quant)])), labels=qvals[quants.index(quant)])
        title_test = f"Residual wrt {quant}"
        # plt.title(title_test)
        plt.xlabel(quant)
        plt.ylabel('Residual')
        outGraph_test = savepath + os.path.sep + f'Model_stability_{quant}_{metric}.png'
        plt.savefig(outGraph_test, bbox_inches='tight')
        plt.clf()
        plt.close()
        i += 1


def visualize_best_models(df, metric="CE"):
    """
        chooses best model based on RMS CrossEntropy metric
        selected for each training type
    """
    model_comparison_metrics = ["CE", "crossentropy", "Dice"]
    assert metric in model_comparison_metrics, ""

    channels = list(df.channel.unique())
    lrs = list(df.lr.unique())
    pts = list(df.pretrained.unique())
    qvaltype = [channels, lrs, pts]
    quants = ['channel', 'lr', 'pretrained']
    i = 0
    best_df_all = get_best_model_df(df, groupby=quants, metric=metric)
    for qvals, quant in zip(qvaltype, quants):
        # print(quant, qvals)
        best_df_quant = get_best_model_df(df, groupby=[quant], metric=metric)
        # print(f"bestdf: {quant}", best_df_quant)
        # for qval in qvals:
        # df_quant = df[df[quant] == qval]
        # best_df = get_best_model_df(df_quant, metric=metric)
        # description_best = f"\n{quant}: {best_df_ch[quant][best_id_ch]}, {best_df_ch['instrument'][best_id_ch]}, {best_df_ch['sampletype'][best_id_ch]} pretrained {best_df_ch['pretrained'][best_id_ch]}@lr{best_df_ch['lr'][best_id_ch]}. Best epoch = {best_df_ch['epoch'][best_id_ch]} "
        # description_best += f"\nmin CE loss= {minloss}, Dice: {best_df_ch['Dice'][best_id_ch]}, F-1 score: {best_df_ch['F1-Score'][best_id_ch]}"
        # print(description_best)
        fig = plt.figure(i)
        if metric in ["CE", "crossentropy"]:
            plt.yscale('log')
        # sns.boxplot(data=best_df_ch, x=quant, y='residual', palette="Spectral", showfliers=False)  # , **props)
        # plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
        sns.stripplot(data=best_df_all, x=quant, y='optimal_metric', color="black", dodge=5, size=3,
                      linewidth=0, legend=False)
        sns.stripplot(data=best_df_quant, x=quant, y='optimal_metric', color="red", dodge=5, size=5,
                      linewidth=0, legend=False)
        do_stat_annotations(df=best_df_all, ax=plt.gca(), pair_param=quant, x=quant, y="optimal_metric", order=None)
        # for (q, v) in zip(best_df_quant[quant], best_df_quant['optimal_metric']):
        for id, row in best_df_quant.iterrows():
            # print( f"{metric}: {row['channel']},\t{row['lr']}\t{row['pretrained']}, \t{(row[quant], (row['optimal_metric']))}")
            plt.gca().annotate(f"{row['optimal_metric']:.4f}\n{row['channel']},\n{row['lr']}\n{row['pretrained']}",
                               (str(row[quant]), (row['optimal_metric'])), xytext=(20, -5), textcoords='offset points',
                               family='sans-serif', fontsize=8, color='darkslategrey')
        # , alpha = 0.5
        # plt.xticks(range(len(qvals[quants.index(quant)])), labels=qvals[quants.index(quant)])
        title_test = f"Best models {quant}"
        # plt.title(title_test)
        plt.xlabel(quant)
        plt.ylabel(f'{metric}')
        outGraph_test = savepath + os.path.sep + f'Best_model_{quant}_{metric}.png'
        plt.savefig(outGraph_test, bbox_inches='tight')
        plt.clf()
        plt.close()
        i += 1


def model_speed(df, use_best=True, metric="CE"):
    """
    Finds the number of epochs for a model that lie in ever smaller squares.
    For this metric, all epochs are taken into account.
    When comparing only the best models, the final best models for each category are identified
    and epochs are used only for those models.

    """
    channels = list(df.channel.unique())
    # print(channels, type(channels))
    thresholds = [5, 4, 3, 2, 1]
    # print(lims, type(lims))
    channel_model_speed = np.zeros((len(channels), len(thresholds)))
    best_df_quant = get_best_model_df(df, metric=metric)
    if use_best:
        df = pd.merge(df, best_df_quant, on=['channel', 'lr', 'pretrained'], suffixes=('', '_best'))
        # print(df)
    for channel in channels:
        df_ch = None
        for lim in thresholds:
            df_ch = df[df['channel'] == channel]
            df_ch = df_ch[df_ch['Train_loss'] <= lim]
            df_ch = df_ch[df_ch['Test_loss'] <= lim]

            channel_model_speed[channels.index(channel), thresholds.index(lim)] = len(df_ch.index)
            # print(channel, lim, len(df_ch.index))
    fig = plt.figure(1)
    plt.plot(channel_model_speed.T)
    plt.gca().invert_xaxis()

    plt.legend(channels, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    title_test = f"Model speed: values below threshold (higher is better)"
    # plt.title(title_test)
    plt.xticks([0, 1, 2, 3, 4], labels=thresholds)
    plt.yticks([0, 200, 400, 600, 800, 1000, 1200], labels=[0, 200, 400, 600, 800, 1000, 1200])
    plt.xlabel('Threshold')
    plt.ylabel('Epochs with values below threshold')
    outGraph_test = savepath + os.path.sep + f'Model_convergence_speed_ch_{metric}{"_best" if use_best else ""}.png'
    plt.savefig(outGraph_test, bbox_inches='tight')
    plt.clf()
    plt.close()
    ###############################################################################################################
    lrs = list(df.lr.unique())
    # print(channels, type(channels))
    thresholds = [5, 4, 3, 2, 1]
    # print(lims, type(lims))
    lr_model_speed = np.zeros((len(lrs), len(thresholds)))
    for lr in lrs:
        df_lr = None
        for lim in thresholds:
            df_lr = df[df['lr'] == lr]
            df_lr = df_lr[df_lr['Train_loss'] <= lim]
            df_lr = df_lr[df_lr['Test_loss'] <= lim]
            lr_model_speed[lrs.index(lr), thresholds.index(lim)] = len(df_lr.index)
    fig = plt.figure(2)
    plt.plot(lr_model_speed.T)
    plt.gca().invert_xaxis()
    plt.legend(lrs, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    title_test = f"Model speed: values below threshold (higher is better)"
    # plt.title(title_test)
    plt.xticks([0, 1, 2, 3, 4], labels=thresholds)
    plt.yticks([0, 200, 400, 600, 800, 1000, 1200], labels=[0, 200, 400, 600, 800, 1000, 1200])
    plt.xlabel('Threshold')
    plt.ylabel('Epochs with values below threshold')
    outGraph_test = savepath + os.path.sep + f'Model_convergence_speed_LR_{metric}{"_best" if use_best else ""}.png'
    plt.savefig(outGraph_test, bbox_inches='tight')
    plt.clf()
    plt.close()
    ###############################################################################################################
    pts = list(df.pretrained.unique())
    # print(channels, type(channels))
    thresholds = [5, 4, 3, 2, 1]
    # print(lims, type(lims))
    pt_model_speed = np.zeros((len(pts), len(thresholds)))
    for pt in pts:
        df_pt = None
        for lim in thresholds:
            df_pt = df[df['pretrained'] == pt]
            df_pt = df_pt[df_pt['Train_loss'] <= lim]
            df_pt = df_pt[df_pt['Test_loss'] <= lim]
            pt_model_speed[pts.index(pt), thresholds.index(lim)] = len(df_pt.index)
    fig = plt.figure(3)
    plt.plot(pt_model_speed.T)
    plt.gca().invert_xaxis()

    plt.legend(pts, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    title_test = f"Model speed: values below threshold (higher is better)"
    # plt.title(title_test)
    plt.xticks([0, 1, 2, 3, 4], labels=thresholds)
    plt.yticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
               labels=[0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    plt.xlabel('Threshold')
    plt.ylabel('Pretrained with values below threshold')
    outGraph_test = savepath + os.path.sep + f'Model_convergence_speed_Pretrained_{metric}{"_best" if use_best else ""}.png'
    plt.savefig(outGraph_test, bbox_inches='tight')
    plt.clf()
    plt.close()
    # return channel_model_speed


if __name__ == "__main__":
    # eval_synthetic = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels/infer_tile_images.csv"
    # eval_measured = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels/inference_Measured.csv"
    # training_data = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels/training.csv"
    # savepath = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels"
    eval_synthetic = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels/MeasuredTrain/infer_tile_images.csv"
    eval_measured = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels/MeasuredTrain/inference_Measured.csv"
    training_data = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels/MeasuredTrain/training.csv"
    savepath = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/CG1D_PS_comparechannels/MeasuredTrain"
    dataF = pd.read_csv(training_data)
    allmetrics = ['Train_loss', 'Test_loss', 'precision', 'recall', 'accuracy', 'F1-Score', 'Dice', 'Jaccard',
                  'MSE']
    ##################################################################
    calcmetrics = True
    if calcmetrics:
        get_stability(dataF, use_best=True, metric="crossentropy")
        for met in ['Dice', 'crossentropy']:
            visualize_best_models(dataF, metric=met)
            # for usebest in [True]:
            model_speed(dataF, use_best=True, metric=met)
    ##################################################################

    visualize = False
    if visualize:
        for i in allmetrics:
            for j in allmetrics:
                if i != j:
                    plot_metrics_comparison(dataF, savepath=savepath, comparemetrics=[i, j], tr_thresh=5, tst_thresh=5)
