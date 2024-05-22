import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_inference_data(df_name, title, savepath, metric='Dice', xaxis_key="channel"):
    df = pd.read_excel(df_name)
    print(df.head())
    idx = df.groupby("channel")["Dice"].idxmax()
    selected_df = df.loc[idx]
    print(selected_df.head())
    print(selected_df.shape)
    selected_df["mixture_tuple"] = selected_df["channel"].apply(
        lambda x: tuple(int(num) for num in x.split("_")[0].split('-')))
    selected_df = selected_df.sort_values(by="mixture_tuple", ascending=False)  # .drop("Sortable", axis=1)
    fig, ax = plt.subplots(1)
    ax.set(ylim=(0, 1.05))
    # print(df.lr.min(), df.lr.max())
    plt.title(title)
    # ax.set(xscale="log")
    sns.stripplot(data=selected_df, x=xaxis_key, hue='lr', y=metric, palette="gnuplot2", linewidth=1.5, s=8)
    plt.legend(bbox_to_anchor=(1.02, 1.02), loc="upper left")
    plt.xticks(rotation=45)
    plt.xlabel("Dataset mixture proportion")
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    # plt.clf()
    # plt.close()
    # return fig


if __name__ == "__main__":
    path = "E:/Data/INFER/PBS/LANL_PBSDDS_Clean_5_10/Combined/"
    dds_smallchecker = "infer_tile_images_cumulative_all.xlsx"
    dds_orig = "infer_tile_images_orig_all.xlsx"
    pbs = "infer_tile_images_pbs_all.xlsx"
    measured = "opposite_evaluated_cumulative_all.xlsx"
    training = "training_dic_best.xlsx"
    inference_dds_small = os.path.join(path, dds_smallchecker)
    # inference_dds_orig = os.path.join(path, dds_orig)
    inference_pbs = os.path.join(path, pbs)
    inference_measured = os.path.join(path, measured)
    validation_train = os.path.join(path, training)
    save_path = "E:/Data/INFER/PBS/LANL_PBSDDS_Clean_5_10/Combined/plots"
    # plot_inference_data(inference_pbs, title="Inference: physics based simulation",
    #                     savepath=f"{save_path}/{os.path.splitext(pbs)[0]}.png")
    # plot_inference_data(inference_measured, title="Inference: measured data",
    #                     savepath=f"{save_path}/{os.path.splitext(measured)[0]}.png")
    plot_inference_data(inference_dds_small, title="Inference: Datadriven simulation",
                        savepath=f"{save_path}/{os.path.splitext(dds_smallchecker)[0]}.png")
    # plot_inference_data(inference_dds_orig, title="Inference: Datadriven simulation",
    #                     savepath=f"{save_path}/{os.path.splitext(dds_orig)[0]}.png")
    plot_inference_data(validation_train, title="Validation: Training",
                        savepath=f"{save_path}/{os.path.splitext(training)[0]}.png")
