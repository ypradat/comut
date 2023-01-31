# -*- coding: utf-8 -*-
"""
@created: Apr 25 2022
@modified: Jan 31 2023
@author: Yoann Pradat

Tests for comut module.
"""

import palettable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from comut import comut

def load_data():
    dfs_data = {}
    dfs_data["mut"] = pd.read_table("examples/tutorial_data/tutorial_mutation_data.tsv")
    dfs_data["pur"] = pd.read_table("examples/tutorial_data/tutorial_purity.tsv")
    dfs_data["bio"] = pd.read_table("examples/tutorial_data/tutorial_biopsy_site.tsv")
    dfs_data["bur"] = pd.read_table("examples/tutorial_data/tutorial_mutation_burden.tsv")
    dfs_data["frq"] = pd.read_table("examples/tutorial_data/tutorial_mutated_samples.tsv")
    dfs_data["frq_bis"] = dfs_data["frq"].copy()
    dfs_data["frq_bis"]["Mutated samples bis"] = 0.5 * dfs_data["frq_bis"]["Mutated samples"]
    dfs_data["sym"] = pd.read_table("examples/tutorial_data/tutorial_symbols.tsv")
    dfs_data["err"] = pd.read_table("examples/tutorial_data/tutorial_error_bars.tsv")

    df_stk = pd.DataFrame({"category": ["BRAF", "NF1", "TP53", "CDKN2A", "NRAS"],
                           "gene 1 M1 tier1": [20, 2, 0, 0, 0],
                           "gene 1 M2 tier1": [2, 0, 0, 0, 0],
                           "gene 1 M1 tier2": [0, 0, 16, 0, 0],
                           "gene 1 M2 tier2": [0, 8, 15, 0, 0],
                           "gene 2 M1 tier2": [3, 12, 5, 0, 0],
                           "None": [25, 28, 14, 50, 50]})

    dfs_data["stk"] = df_stk

    return dfs_data


def load_mapping():
    vivid_10 = palettable.cartocolors.qualitative.Vivid_10.mpl_colors
    purp_7 = palettable.cartocolors.sequential.Purp_7.mpl_colormap

    mappings = {}
    mappings["mut"] = {'Missense': vivid_10[5], 'Nonsense': vivid_10[0], 'In frame indel': vivid_10[1],
                       'Frameshift indel': vivid_10[4], 'Splice site': vivid_10[9]}
    mappings["pur"] = purp_7
    mappings["bio"] = {"Kidney": vivid_10[2], "Liver": vivid_10[4], "Lung": vivid_10[6], "Lymph Node": vivid_10[8]}
    mappings["bur"] = {'Nonsynonymous': vivid_10[8], 'Synonymous':purp_7(0.5)}
    mappings["frq"] = {"Mutated samples": "lightgrey"}
    mappings["frq_bis"] = {"Mutated samples": "gold", "Mutated samples bis": "blue"}
    mappings["sym"] = {"resistance": ("black", "red"), "sensitivity": ("gold", "gold")}
    mappings["err"] = {"tier1": [0.2, 'limegreen', 'palegreen', '-', 2, 'o', 6],
                       "tier2": [-0.2, 'royalblue', 'lightskyblue', '-', 2, 'o', 6]}
    mappings["stk"] = {"gene 1 M1 tier1": vivid_10[0], "gene 1 M1 tier2": vivid_10[0],
                       "gene 1 M2 tier1": vivid_10[3], "gene 1 M2 tier2": vivid_10[3],
                       "gene 2 M1 tier2": vivid_10[6], "None": "lightgrey"}
    mappings["stk_edges"] = {"tier1": 'limegreen', "tier2": 'royalblue', "None": None}

    return mappings


def render_plot(comut_object, figsize=(10,3), x_padding=0.04, y_padding=0.04, tri_padding=0.03,
                hspace=0.2, wspace=0.2, **kwargs):
    comut_object.plot_comut(x_padding=x_padding,
                            y_padding=y_padding,
                            tri_padding=tri_padding,
                            figsize=figsize,
                            hspace=hspace,
                            wspace=wspace,
                            **kwargs)


def save_plot(comut_object, filepath, dpi=300):
    fig = comut_object.figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print("-plot saved at %s" % filepath)


def test_comut_simple():
    dfs_data = load_data()
    mappings = load_mapping()

    # columns ordering
    cols = ["Sample%d" % i for i in range(1,51)]

    # rows ordering
    rows = ['CDKN2A', 'TP53', 'NF1', 'NRAS', 'BRAF']

    comut_test = comut.CoMut()
    comut_test.samples = cols
    comut_test.add_categorical_data(data=dfs_data["mut"], name='Mutation type', category_order=rows,
                                    mapping=mappings["mut"], xtick_show=True,
                                    ytick_style='normal', ytick_fontdict={"fontsize": 12})

    render_plot(comut_object=comut_test)
    comut_test.add_unified_legend()
    save_plot(comut_test, filepath="tests/plots/comut_simple.svg")


def test_comut_design_1():
    dfs_data = load_data()
    mappings = load_mapping()

    # columns ordering
    cols = ["Sample%d" % i for i in range(1,51)]

    # rows ordering
    rows = ['CDKN2A', 'TP53', 'NF1', 'NRAS', 'BRAF']

    comut_test = comut.CoMut()
    comut_test.samples = cols
    comut_test.add_categorical_data(data=dfs_data["mut"], name='Mutation type', category_order=rows,
                                    mapping=mappings["mut"], xtick_show=True, xtick_fontdict={"fontsize": 8},
                                    ytick_style='italic', ytick_fontdict={"fontsize": 12})


    comut_test.add_continuous_data(data=dfs_data["pur"], name='Purity',
                                   mapping=mappings["pur"], xtick_show=False,
                                   ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_categorical_data(data=dfs_data["bio"], name='Biopsy site',
                                    mapping=mappings["bio"], xtick_show=False,
                                    ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_bar_data(data=dfs_data["bur"], name='Mutation burden',
                            mapping=mappings["bur"], ytick_fontdict={"fontsize": 12}, stacked=True,
                            ylabel="Nosynon.\nMutations", ylabel_rotation=90)


    comut_test.add_side_bar_data(data=dfs_data["frq"], paired_name='Mutation type', name="Mutated samples",
                                 position="left", mapping=mappings["frq"], xtick_fontdict={"fontsize": 12},
                                 stacked=True, xlabel="Mutated samples", xlabel_rotation=0)

    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.25)
    comut_test.add_unified_legend(ncol=2)
    save_plot(comut_test, filepath="tests/plots/comut_design_1.svg")


def test_comut_design_2():
    dfs_data = load_data()
    mappings = load_mapping()

    # columns ordering
    cols = ["Sample%d" % i for i in range(1,51)]

    # rows ordering
    rows = ['CDKN2A', 'TP53', 'NF1', 'NRAS', 'BRAF']

    comut_test = comut.CoMut()
    comut_test.samples = cols
    comut_test.add_categorical_data(data=dfs_data["mut"], name='Mutation type', category_order=rows,
                                    mapping=mappings["mut"], xtick_show=True, xtick_fontdict={"fontsize": 8},
                                    ytick_style='italic', ytick_fontdict={"fontsize": 12})


    comut_test.add_scatter_data(data=dfs_data["sym"], name='Resistances', paired_name='Mutation type',
                                mapping=mappings["sym"], marker="*", markersize=10)


    comut_test.add_continuous_data(data=dfs_data["pur"], name='Purity',
                                   mapping=mappings["pur"], xtick_show=False,
                                   ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_categorical_data(data=dfs_data["bio"], name='Biopsy site',
                                    mapping=mappings["bio"], xtick_show=False,
                                    ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_bar_data(data=dfs_data["bur"], name='Mutation burden',
                            mapping=mappings["bur"], ytick_fontdict={"fontsize": 12}, stacked=True,
                            ylabel="Nosynon.\nMutations", ylabel_rotation=90)


    comut_test.add_side_bar_data(data=dfs_data["frq"], paired_name='Mutation type', name="Mutated samples",
                                 position="left", mapping=mappings["frq"], xtick_fontdict={"fontsize": 12},
                                 stacked=True, xlabel="Mutated samples", xlabel_rotation=0)

    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.25)
    comut_test.add_unified_legend(ncol=2)

    # edit individual axes
    comut_test.axes['Mutation type'].set_xticklabels([])

    # calculate the percentage of samples with that gene mutated, rounding and adding a percent sign
    df_freq = dfs_data["frq"]
    df_freq = df_freq.set_index("category").loc[rows].reset_index()
    percentages = (df_freq['Mutated samples']/len(comut_test.samples)*100).round(1).astype(str) + '%'

    # set location of yticks
    comut_test.axes['Mutated samples'].set_yticks(np.arange(0.5, len(rows)+0.5))

    # set labels of yticks
    comut_test.axes['Mutated samples'].set_yticklabels(list(percentages), ha="right")

    # move the ytick labels inside the bar graph
    comut_test.axes['Mutated samples'].tick_params(axis='y', pad=0, length=0)

    # Make y axis visible (by default it is not visible)
    comut_test.axes['Mutated samples'].get_yaxis().set_visible(True)

    # move y axis ticks to the right
    comut_test.axes['Mutated samples'].yaxis.tick_right()

    save_plot(comut_test, filepath="tests/plots/comut_design_2.svg")


def test_comut_design_3():
    dfs_data = load_data()
    mappings = load_mapping()

    # columns ordering
    cols = ["Sample%d" % i for i in range(1,51)]

    # rows ordering
    rows = ['CDKN2A', 'TP53', 'NF1', 'NRAS', 'BRAF']

    comut_test = comut.CoMut()
    comut_test.samples = cols
    comut_test.add_categorical_data(data=dfs_data["mut"], name='Mutation type', category_order=rows,
                                    mapping=mappings["mut"], xtick_show=True, xtick_fontdict={"fontsize": 8},
                                    ytick_style='italic', ytick_fontdict={"fontsize": 12})


    comut_test.add_continuous_data(data=dfs_data["pur"], name='Purity',
                                   mapping=mappings["pur"], xtick_show=False,
                                   ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_categorical_data(data=dfs_data["bio"], name='Biopsy site',
                                    mapping=mappings["bio"], xtick_show=False,
                                    ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_bar_data(data=dfs_data["bur"], name='Mutation burden',
                            mapping=mappings["bur"], ytick_fontdict={"fontsize": 12}, stacked=True,
                            ylabel="Nosynon.\nMutations", ylabel_rotation=90)


    comut_test.add_side_bar_data(data=dfs_data["frq"], paired_name='Mutation type', name="Mutated samples",
                                 position="left", mapping=mappings["frq"], xtick_fontdict={"fontsize": 12},
                                 stacked=True, xlabel="Mutated samples", xlabel_rotation=0)

    comut_test.add_side_bar_data(data=dfs_data["frq"], paired_name='Mutation type', name="Mutated samples bis",
                                 position="right", mapping=mappings["frq"], xtick_fontdict={"fontsize": 12},
                                 stacked=True, xlabel="Mutated samples", xlabel_rotation=0)

    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.1, widths=[1,5,1], shadow_width_left=0.7)
    comut_test.add_unified_legend(ncol=2, axis_name="Mutated samples bis")
    save_plot(comut_test, filepath="tests/plots/comut_design_3.svg")


def test_comut_design_4():
    dfs_data = load_data()
    mappings = load_mapping()

    # columns ordering
    cols = ["Sample%d" % i for i in range(1,51)]

    # rows ordering
    rows = ['CDKN2A', 'TP53', 'NF1', 'NRAS', 'BRAF']

    comut_test = comut.CoMut()
    comut_test.samples = cols
    comut_test.add_categorical_data(data=dfs_data["mut"], name='Mutation type', category_order=rows,
                                    mapping=mappings["mut"], xtick_show=True, xtick_fontdict={"fontsize": 8},
                                    ytick_style='italic', ytick_fontdict={"fontsize": 12})


    comut_test.add_continuous_data(data=dfs_data["pur"], name='Purity',
                                   mapping=mappings["pur"], xtick_show=False,
                                   ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_categorical_data(data=dfs_data["bio"], name='Biopsy site',
                                    mapping=mappings["bio"], xtick_show=False,
                                    ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_bar_data(data=dfs_data["bur"], name='Mutation burden',
                            mapping=mappings["bur"], ytick_fontdict={"fontsize": 12}, stacked=True,
                            ylabel="Nosynon.\nMutations", ylabel_rotation=90)


    comut_test.add_side_bar_data(data=dfs_data["frq"], paired_name='Mutation type', name="Mutated samples",
                                 position="left", mapping=mappings["frq"], xtick_fontdict={"fontsize": 12},
                                 stacked=True, xlabel="Mutated samples", xlabel_rotation=0)

    comut_test.add_side_error_data(data=dfs_data["err"], paired_name='Mutation type', name="Odds ratio",
                                   position="right", mapping=mappings["err"], xtick_fontdict={"fontsize": 10},
                                   xlabel="Odds ratio", xlabel_rotation=0)


    # replace legend of Mutation type
    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.1, widths=[1,5,1], shadow_width_left=0.7)

    green_star = mlines.Line2D([], [], color='limegreen', marker='*', linestyle='None', markersize=10)
    blue_star = mlines.Line2D([], [], color='royalblue', marker='*', linestyle='None', markersize=10)
    handles = [green_star, blue_star]
    labels = ["Tier1", "Tier2"]

    handles_more = [handles]
    labels_more = [labels]
    titles_more = ["Resistances"]

    comut_test.add_unified_legend(ncol=2, axis_name="Odds ratio", ignored_plots=["Mutation type"],
                                  handles_more=handles_more, labels_more=labels_more, titles_more=titles_more)

    comut_test.axes['Odds ratio'].axvline(1, color = 'black', linestyle = 'dotted', linewidth = 2)
    comut_test.axes['Odds ratio'].set_xscale('log')
    comut_test.axes['Odds ratio'].set_xticks([0.2, 1, 5])
    comut_test.axes['Odds ratio'].set_xticklabels([0.2, 1, 5])

    save_plot(comut_test, filepath="tests/plots/comut_design_4.svg")


def test_comut_design_5():
    dfs_data = load_data()
    mappings = load_mapping()

    # columns ordering
    cols = ["Sample%d" % i for i in range(1,51)]

    # rows ordering
    rows = ['CDKN2A', 'TP53', 'NF1', 'NRAS', 'BRAF']

    comut_test = comut.CoMut()
    comut_test.samples = cols
    comut_test.add_categorical_data(data=dfs_data["mut"], name='Mutation type', category_order=rows,
                                    mapping=mappings["mut"], xtick_show=True, xtick_fontdict={"fontsize": 8},
                                    ytick_style='italic', ytick_fontdict={"fontsize": 12})


    comut_test.add_continuous_data(data=dfs_data["pur"], name='Purity',
                                   mapping=mappings["pur"], xtick_show=False,
                                   ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_categorical_data(data=dfs_data["bio"], name='Biopsy site',
                                    mapping=mappings["bio"], xtick_show=False,
                                    ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_bar_data(data=dfs_data["bur"], name='Mutation burden',
                            mapping=mappings["bur"], ytick_fontdict={"fontsize": 12}, stacked=True,
                            ylabel="Nosynon.\nMutations", ylabel_rotation=90)


    comut_test.add_side_bar_data(data=dfs_data["frq"], paired_name='Mutation type', name="Mutated samples",
                                 position="left", mapping=mappings["frq"], xtick_fontdict={"fontsize": 12},
                                 stacked=True, xlabel="Mutated samples", xlabel_rotation=0)

    comut_test.add_side_error_data(data=dfs_data["err"], paired_name='Mutation type', name="Odds ratio",
                                   position="right", mapping=mappings["err"], xtick_fontdict={"fontsize": 10},
                                   xlabel="Odds ratio", xlabel_rotation=0)

    comut_test.add_side_bar_data(data=dfs_data["stk"], paired_name='Mutation type', name="Stacked bars", stacked=True,
                                 position="right", mapping=mappings["stk"], xtick_fontdict={"fontsize": 10},
                                 xlabel="Alterations\nconferring resistance", xlabel_fontsize=10, xlabel_rotation=0)


    # replace legend of Mutation type
    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.1, widths=[1,5,1,1.5], shadow_width_left=0.7)

    handles_more = []
    labels_more = []
    titles_more = []

    # legend for stars
    green_star = mlines.Line2D([], [], color='limegreen', marker='*', linestyle='None', markersize=10)
    blue_star = mlines.Line2D([], [], color='royalblue', marker='*', linestyle='None', markersize=10)
    handles = [green_star, blue_star]
    labels = ["Tier1", "Tier2"]


    handles_more.append(handles)
    labels_more.append(labels)
    titles_more.append("Resistances")

    # legend for stacked bars
    labels = [label for label, _ in mappings["stk"].items()]
    colors = [color for _, color in mappings["stk"].items()]
    df_lc = pd.DataFrame({"label": labels, "color": colors})
    df_lc["label"] = df_lc["label"].apply(lambda x: x.split("tier")[0].strip())
    df_lc = df_lc.drop_duplicates(subset=["label"], keep="first")

    handles = [mpatches.Patch(color=color) for color in df_lc["color"]]
    labels = [label for label in df_lc["label"]]

    handles_more.append(handles)
    labels_more.append(labels)
    titles_more.append("Alterations")

    comut_test.add_unified_legend(ncol=2, axis_name="Stacked bars", ignored_plots=["Mutation type"],
                                  handles_more=handles_more, labels_more=labels_more, titles_more=titles_more)

    # edit individual axes
    axis_name = "Stacked bars"

    # set correct order for bars
    df_stk = dfs_data["stk"].copy()
    df_stk_indexed = df_stk.set_index('category')
    df_stk_indexed = df_stk_indexed.reindex(comut_test._plots["Mutation type"]["data"].index)
    df_stk_edges = pd.DataFrame(index=df_stk_indexed.index)
    patterns = ["tier1", "tier2", "None"]
    for pattern in patterns:
        cols_pattern = [x for x in df_stk_indexed if x.endswith(pattern)]
        df_stk_edges[pattern] = df_stk_indexed[cols_pattern].sum(axis=1)

    y_range = np.arange(0.5, len(df_stk_edges.index))
    cum_bar_df = np.cumsum(df_stk_edges, axis=1)

    # for each bar, calculate bottom and top of bar and plot
    for i in range(len(cum_bar_df.columns)):
        column = cum_bar_df.columns[i]
        color = mappings["stk_edges"][column]
        if color is not None:
            if i == 0:
                left = None
                bar_data = cum_bar_df.loc[:, column]
            else:
                # calculate distance between previous and current column
                prev_column = cum_bar_df.columns[i-1]
                bar_data = cum_bar_df.loc[:, column] - cum_bar_df.loc[:, prev_column]

                # previous column defines the "bottom" of the bars
                left = cum_bar_df.loc[:, prev_column]

            # mask not zero
            mask_nz = bar_data!=0
            if sum(mask_nz) > 0:
                y_range_nz = y_range[mask_nz]
                bar_data_nz = bar_data[mask_nz]

                if left is not None:
                    left_nz = left[mask_nz]
                else:
                    left_nz = None

                comut_test.axes[axis_name].barh(y_range_nz, bar_data_nz, align='center', facecolor='none',
                                                edgecolor=color, lw=2, left=left_nz)

    save_plot(comut_test, filepath="tests/plots/comut_design_5.svg")


def test_comut_design_6():
    dfs_data = load_data()
    mappings = load_mapping()

    # columns ordering
    cols = ["Sample%d" % i for i in range(1,51)]

    # rows ordering
    rows = ['CDKN2A', 'TP53', 'NF1', 'NRAS', 'BRAF']

    comut_test = comut.CoMut()
    comut_test.samples = cols
    comut_test.add_categorical_data(data=dfs_data["mut"], name='Mutation type', category_order=rows,
                                    mapping=mappings["mut"], xtick_show=True, xtick_fontdict={"fontsize": 8},
                                    ytick_style='italic', ytick_fontdict={"fontsize": 12})


    comut_test.add_continuous_data(data=dfs_data["pur"], name='Purity',
                                   mapping=mappings["pur"], xtick_show=False,
                                   ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_categorical_data(data=dfs_data["bio"], name='Biopsy site',
                                    mapping=mappings["bio"], xtick_show=False,
                                    ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_bar_data(data=dfs_data["bur"], name='Mutation burden',
                            mapping=mappings["bur"], ytick_fontdict={"fontsize": 12}, stacked=True,
                            ylabel="Nosynon.\nMutations", ylabel_rotation=90)


    comut_test.add_side_bar_data(data=dfs_data["frq_bis"], paired_name='Mutation type', name="Mutated samples",
                                 position="left", mapping=mappings["frq_bis"], xtick_fontdict={"fontsize": 12},
                                 stacked=False, xlabel="Mutated samples", xlabel_rotation=0)

    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.25)
    comut_test.add_unified_legend(ncol=2, labels_orders={"Biopsy site": ["Lung", "Lymph Node", "Liver", "Kidney"]})
    save_plot(comut_test, filepath="tests/plots/comut_design_6.svg")


def test_comut_design_7():
    dfs_data = load_data()
    mappings = load_mapping()

    # columns ordering
    cols = ["Sample%d" % i for i in range(1,51)]

    # rows ordering
    rows = ['CDKN2A', 'TP53', 'NF1', 'NRAS', 'BRAF']

    comut_test = comut.CoMut()
    comut_test.samples = cols
    comut_test.add_categorical_data(data=dfs_data["mut"], name='Mutation type', category_order=rows,
                                    mapping=mappings["mut"], xtick_show=True, xtick_fontdict={"fontsize": 8},
                                    ytick_style='italic', ytick_fontdict={"fontsize": 12})


    comut_test.add_continuous_data(data=dfs_data["pur"], name='Purity',
                                   mapping=mappings["pur"], xtick_show=False,
                                   ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_categorical_data(data=dfs_data["bio"], name='Biopsy site',
                                    mapping=mappings["bio"], xtick_show=False,
                                    ytick_style='normal', ytick_fontdict={"fontsize": 12})


    comut_test.add_bar_data(data=dfs_data["bur"], name='Mutation burden',
                            mapping=mappings["bur"], ytick_fontdict={"fontsize": 12}, stacked=True,
                            ylabel="Nosynon.\nMutations", ylabel_rotation=90)


    comut_test.add_side_bar_data(data=dfs_data["frq"], paired_name='Mutation type', name="Mutated samples",
                                 position="left", mapping=mappings["frq"], xtick_fontdict={"fontsize": 12},
                                 stacked=True, xlabel="Mutated samples", xlabel_rotation=0)

    comut_test.add_side_error_data(data=dfs_data["err"], paired_name='Mutation type', name="Odds ratio",
                                   position="right", mapping=mappings["err"], xtick_fontdict={"fontsize": 10},
                                   xlabel="Odds ratio", xlabel_rotation=0)


    # replace legend of Mutation type
    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.1, widths=[1,5,1], shadow_width_left=0.7)

    green_star = mlines.Line2D([], [], color='limegreen', marker='*', linestyle='None', markersize=10)
    blue_star = mlines.Line2D([], [], color='royalblue', marker='*', linestyle='None', markersize=10)
    handles = [green_star, blue_star]
    labels = ["Tier1", "Tier2"]

    handles_more = [handles]
    labels_more = [labels]
    titles_more = ["Resistances"]

    comut_test.add_unified_legend(nrow=5, axis_name="Odds ratio", ignored_plots=["Mutation type"],
                                  handles_more=handles_more, labels_more=labels_more, titles_more=titles_more,
                                  headers_separate=["Mutation burden", "Biopsy site", "Resistances"])

    comut_test.axes['Odds ratio'].axvline(1, color = 'black', linestyle = 'dotted', linewidth = 2)
    comut_test.axes['Odds ratio'].set_xscale('log')
    comut_test.axes['Odds ratio'].set_xticks([0.2, 1, 5])
    comut_test.axes['Odds ratio'].set_xticklabels([0.2, 1, 5])

    save_plot(comut_test, filepath="tests/plots/comut_design_7.svg")
