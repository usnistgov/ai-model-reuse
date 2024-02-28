import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns

xaxis_key = 'checker size'
# xaxis_key = 'dataset size (images)'


def visualize(df, title="Title", metric='Dice', lrmid=False):
    print(df.columns)
    if lrmid == "mid":
        df = df[(df['lr'] == 1e-3) | (df['lr'] == 1e-2) | (df['lr'] == 1e-1)]
    if isinstance(lrmid, list):
        df = df[df['lr'].isin(lrmid)]
        # dflist = []
        # for lrval in lrmid:
        #     dflist.append(df['lr'])
    fig, ax = plt.subplots(1)

    ax.set(ylim=(0, 1.05))

    props = {
        'boxprops': {'facecolor': 'green', 'edgecolor': 'black'},
        # 'boxprops': {'edgecolor': 'Spectral'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'green'},
        'capprops': {'color': 'green'}
    }
    print(df.lr.min(), df.lr.max())
    plt.title(title)
    # ax.set(xscale="log")
    sns.stripplot(data=df, x=xaxis_key, hue='lr', y=metric, palette="gnuplot2", linewidth=1.5, s=8)
    # if not lrmid:
    #     sns.boxplot(data=df, x=xaxis_key, y=metric, **props)
    # plt.xticks(df['lr'].unique())

    plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    return fig


def visualizelr(df, title="Title", metric='Dice', lrmid=False):
    print(df.columns)
    # if lrmid == "mid":
    #     print(df['lr'][2]==1e-4)
    #     df = df[(df['lr'] == 1e-3) | (df['lr'] == 1e-4)| (df['lr'] == 1e-6)]
    fig, ax = plt.subplots(1)
    ax.set(ylim=(0, 1.05))

    props = {
        'boxprops': {'facecolor': 'green', 'edgecolor': 'black'},
        # 'boxprops': {'edgecolor': 'Spectral'},
        'medianprops': {'color': 'black'},
        'whiskerprops': {'color': 'green'},
        'capprops': {'color': 'green'}
    }
    print(df.lr.min(), df.lr.max())
    plt.title(title)

    ax.set(ylim=(0, 1.05))
    sns.stripplot(data=df, hue=xaxis_key, x='lr', y=metric, palette="gnuplot2", linewidth=1.5, s=8)
    # sns.boxplot(data=df, x='lr', y=metric, **props)
    # plt.xticks(df['lr'].unique())

    plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    return fig


def visualize_best_lr(df, title="Title", labelwise=False, lrmid=False):
    if lrmid == "mid":
        df = df[(df['lr'] == 1e-3) | (df['lr'] == 1e-2) | (df['lr'] == 1e-1)]
    if isinstance(lrmid, list):
        df = df[df['lr'].isin(lrmid)]
    print("ncheck", df)
    fig, ax = plt.subplots(1)

    ax.set(ylim=(0, 1.05))
    plt.title(title)
    if labelwise:
        dice_cols = [col for col in df.columns if 'Dice_' in col]
        colors = plt.cm.gist_ncar(np.linspace(0, 1, 15))
        markers = [".", ",", "o", "v", "^", "<", ">", "s", "p", "*", "D", "X", "P", "1", "d"]
        for i, dice_col in enumerate(dice_cols):
            max_values = df.groupby(['lr'])[dice_cols].max().reset_index()
            print(max_values.columns)
            plt.plot(max_values["lr"].astype(str), max_values[dice_col], color=colors[i], marker=markers[i],
                     label=dice_col, linestyle='-')
        plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    else:
        df = df.sort_values(by="Dice", ascending=False).groupby('lr').head(1)
        sns.stripplot(data=df, x='lr', y="Dice", palette="gnuplot2", linewidth=1.5, s=8)
    return fig


def visualize_best_ncheck(df, title="Title", labelwise=False, lrmid=False):
    if lrmid == "mid":
        df = df[(df['lr'] == 1e-3) | (df['lr'] == 1e-2) | (df['lr'] == 1e-1)]
    if isinstance(lrmid, list):
        df = df[df['lr'].isin(lrmid)]
    print("lr\n", df)
    fig, ax = plt.subplots(1)

    ax.set(ylim=(0, 1.05))
    plt.title(title)
    colors = plt.cm.gist_ncar(np.linspace(0, 1, 15))
    markers = [".", ",", "o", "v", "^", "<", ">", "s", "p", "*", "D", "X", "P", "1", "d"]
    if labelwise:
        dice_cols = [col for col in df.columns if 'Dice_' in col]
        for i, dice_col in enumerate(dice_cols):
            max_values = df.groupby([xaxis_key])[dice_cols].max().reset_index()
            plt.plot(max_values[xaxis_key], max_values[dice_col], color=colors[i], marker=markers[i],
                     label=dice_col, linestyle='-')
        plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    else:
        df = df.sort_values(by="Dice", ascending=False).groupby(xaxis_key).head(1)
        sns.stripplot(data=df, x=xaxis_key, y="Dice", palette="gnuplot2", linewidth=1.5, s=8)
    # plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    return fig


def read_and_plot(path, savename):
    df = pd.read_excel(path, index_col=0)

    # lrmid = True
    vals = [1e-3]  # Add selected numbers here
    for lrmid in ["mid", None, vals]:
    # for lrmid in ["mid", None]:
        lrm = ""
        if lrmid == "mid":
            lrm = "_lrmid"
        if isinstance(lrmid, list):
            lrm = "_selected"
        savepathxcheck = os.path.dirname(path) + os.path.sep + f'{savename}{lrm}_ncheck.png'
        savename = savename.replace("_", " ")
        visualize(df, title=savename, lrmid=lrmid)
        plt.savefig(savepathxcheck, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        savepathxlr = os.path.dirname(path) + os.path.sep + f'{savename}{lrm}_lr.png'
        visualizelr(df, title=savename, lrmid=lrmid)
        plt.savefig(savepathxlr, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        savepath_ldice_lr = os.path.dirname(path) + os.path.sep + f'{savename}{lrm}_lr_labelwise.png'
        visualize_best_lr(df, title=savename, lrmid=lrmid, labelwise=True)
        plt.savefig(savepath_ldice_lr, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        savepath_ldice_ncheck = os.path.dirname(path) + os.path.sep + f'{savename}{lrm}_ncheck_labelwise.png'
        visualize_best_ncheck(df, title=savename, lrmid=lrmid, labelwise=True)
        plt.savefig(savepath_ldice_ncheck, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        savepathbestxlr = os.path.dirname(path) + os.path.sep + f'{savename}{lrm}_lr_best.png'
        visualize_best_lr(df, title=savename, lrmid=lrmid)
        plt.savefig(savepathbestxlr, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()

        savepathbestxncheck = os.path.dirname(path) + os.path.sep + f'{savename}{lrm}_ncheck_best.png'
        visualize_best_ncheck(df, title=savename, lrmid=lrmid)
        plt.savefig(savepathbestxncheck, bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()


def main():
    if xaxis_key == "checker size":
        synthetic_inference = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/ncheck_CG1D_PS_size/infer_tile_images_cumulative_size_all.xlsx"
        synthetic_inference_orig = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/ncheck_CG1D_PS_size/infer_tile_images_cumulative_orig_size_all.xlsx"
        measured_inference = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/ncheck_CG1D_PS_size/opposite_evaluated_cumulative_size_all.xlsx"
        read_and_plot(synthetic_inference_orig, 'Synthetic_inference_original')
    elif xaxis_key == "dataset size (images)":
        synthetic_inference = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/ncheck_CG1D_PS/infer_tile_images_cumulative_all.xlsx"
        measured_inference = "C:/Users/pss2/NetBeansProjects/stats-simulations/data/ncheck_CG1D_PS/opposite_evaluated_cumulative_all.xlsx"
    read_and_plot(synthetic_inference, 'Synthetic_inference')
    read_and_plot(measured_inference, 'Measured_inference')


if __name__ == "__main__":
    main()
