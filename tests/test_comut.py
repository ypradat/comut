# -*- coding: utf-8 -*-
"""
@modified: Apr 14 2022
@created: Apr 14 2022
@author: Yoann Pradat

Tests for comut module.
"""

import palettable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from comut import comut

def load_data():
    dfs_data = {}
    dfs_data["mut"] = pd.read_table("examples/tutorial_data/tutorial_mutation_data.tsv")
    dfs_data["pur"] = pd.read_table("examples/tutorial_data/tutorial_purity.tsv")
    dfs_data["bio"] = pd.read_table("examples/tutorial_data/tutorial_biopsy_site.tsv")
    dfs_data["bur"] = pd.read_table("examples/tutorial_data/tutorial_mutation_burden.tsv")
    dfs_data["frq"] = pd.read_table("examples/tutorial_data/tutorial_mutated_samples.tsv")
    dfs_data["sym"] = pd.read_table("examples/tutorial_data/tutorial_symbols.tsv")

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
    mappings["frq"] = {"Mutated samples": "darkgrey"}
    mappings["sym"] = {"resistance": "black", "sensitivity": "gold"}

    return mappings


def render_plot(comut_object, figsize=(10,3), x_padding=0.04, y_padding=0.04, tri_padding=0.03,
                hspace=0.2, wspace=0.2, legend_ncol=1, **kwargs):
    comut_object.plot_comut(x_padding=x_padding,
                            y_padding=y_padding,
                            tri_padding=tri_padding,
                            figsize=figsize,
                            hspace=hspace,
                            wspace=wspace,
                            **kwargs)
    comut_object.add_unified_legend(ncol=legend_ncol)


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


    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.25)
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

    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.25, legend_ncol=2)
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


    comut_test.add_scatter_data(data=dfs_data["sym"], name='Resistances', paired_name='Mutation type',
                                mapping=mappings["sym"], marker="*", s=25)


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

    render_plot(comut_object=comut_test, hspace=0.1, wspace=0.25, legend_ncol=2)

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

    save_plot(comut_test, filepath="tests/plots/comut_design_3.svg")
