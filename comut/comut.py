from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.offsetbox as offsetbox
import palettable
from collections import defaultdict


class CoMut:
    '''A user-created :class: `CoMut` object.

    Params:
    -----------

    Attributes:
    -----------
    samples: list
        List of samples that defines the sample order. It is set
        by the first data set added. Samples from later data sets
        are checked and reordered against this attribute.

    axes: dict
        Container containing plotted axes objects after plot_comut()
        is called. Axes objects can be accessed and changed to change
        the CoMut.

    figure: matplotlib figure object
        Figure that the CoMut is plotted on

    _plots: dict
        Container for plot information, including data, visual
        params (ie color maps), plot type, and plot name.

    _side_plots: dict of dicts
        Container for side plot information. Values are side plot
        data, keys are the name of the central CoMut plot the side
        plot is paired with.'''

    def __init__(self):

        # user accessible attributes
        self.samples = None
        self.genes = None
        self.axes = {}
        self.figure = None

        # attributes for manipulation and storage
        self._plots = {}
        self._side_plots = defaultdict(dict)
        self._over_plots = defaultdict(dict)

    @classmethod
    def _get_default_categorical_cmap(cls, n_cats):
        '''Returns the default color map for n categories.
        If 10 or fewer, uses vivid_10 from palettable. If more than 10,
        uses a segmented rainbow colormap.

        Params:
        -------
        n_cats: int
            The number of categories in the data.

        Returns:
        --------
        cmap: list of colors'''

        if n_cats <= 10:
            cmap = palettable.cartocolors.qualitative.Vivid_10.mpl_colors
        else:
            hsv_cmap = plt.get_cmap('hsv')
            cmap = [hsv_cmap(i/n_cats) for i in range(n_cats)]

        return cmap

    @classmethod
    def _get_triangles(cls, x_base, y_base, tri_padding, height, width):
        '''Returns np arrays of triangle coordinates

        Params:
        -------
        x_base, y_base: floats
            The x and y coordinates of the base of the triangle

        tri_padding: float
            The space between triangles

        height, width: float
            Height and width of the box enclosing the triangles.

        Returns:
        --------
        (tri_1_coords, tri_2_coords): tuple of np arrays
            Tuple of triangle coordinates as np arrays.'''

        tri_1_coords = [[x_base, y_base + tri_padding],
                        [x_base, y_base + height],
                        [x_base + width - tri_padding, y_base + height]]

        tri_2_coords = [[x_base + tri_padding, y_base],
                        [x_base + width, y_base],
                        [x_base + width, y_base + height - tri_padding]]

        return (np.array(tri_1_coords), np.array(tri_2_coords))

    @classmethod
    def _sort_list_by_list(cls, value_list, value_order):
        '''Sort an value list by a specified value
        order, otherwise sort alphabetically at end.

        Params:
        -------
        value_list: list-like
            values to sort, eg ['nonsense', 'amp']

        value_order: list-like
            List of values that specify sort order.
            Values not in this list will be sorted alphabetically
            and placed at the end of the list.

        Returns:
        --------
        sorted_values: list
            Values sorted by the value order specified and
            alphabetically otherwise.'''

        # extract subset of alts that are specified in value_order
        subset = [value for value in value_list if value in value_order]
        other = [value for value in value_list if value not in value_order]

        # sort subset according to value order list, otherwise alphabetical
        sorted_subset = sorted(subset, key=lambda x: value_order.index(x))
        sorted_other = sorted(other)

        # join the two subsets
        sorted_values = sorted_subset + sorted_other
        return sorted_values

    @classmethod
    def _parse_categorical_data(cls, data, category_order, sample_order,
                                value_order, priority, borders):
        '''Parses tidy dataframe into a gene x sample dataframe
        of tuples for plotting

        Params:
        -------
        data: pandas dataframe
            Dataframe from add_categorical_data or add_continuous_data

        category_order: list-like
            category_order from add_categorical_data

        sample_order: list-like
            Order of samples, from left to right.

        value_order: list-like:
            value_order from add_categorical_data

        priority: list-like
            priority from add_categorical_data

        borders: list-like
            borders from add_categorical_data

        Returns:
        --------
        parsed_data: pandas dataframe, shape (categories, samples)
            Dataframe of tuples depicting values for each sample in
            each category.'''

        # create parsed data storage
        parsed_data = pd.DataFrame(index=category_order, columns=sample_order)

        # subset data to categories and samples to avoid handling large dataframes
        data = data[(data['category'].isin(category_order)) &
                    (data['sample'].isin(sample_order))]

        # fill in parsed dataframe
        for category in category_order:
            for sample in sample_order:
                sample_category_data = data[(data['category'] == category) &
                                            (data['sample'] == sample)]

                # if data is empty, the sample does not have a value in category
                if len(sample_category_data) == 0:
                    parsed_data.loc[category, sample] = ()

                # if length 1 just put the value
                elif len(sample_category_data) == 1:
                    value = sample_category_data['value'].values[0]
                    parsed_data.loc[category, sample] = (value,)

                # if length 2, sort by value order then convert to tuple
                elif len(sample_category_data) == 2:
                    values = sample_category_data['value'].values
                    sorted_values = cls._sort_list_by_list(values, value_order)
                    parsed_data.loc[category, sample] = tuple(sorted_values)

                # if more than two, apply priority, sort, then convert to tuple.
                else:
                    values = sample_category_data['value'].values
                    present_priorities = [v for v in values if v in priority]
                    present_borders = [v for v in values if v in borders]

                    # just put 'Multiple' if no priorities or more than two
                    if len(present_priorities) == 0 or len(present_priorities) > 2:
                        parsed_data.loc[category, sample] = tuple(['Multiple'] + present_borders)

                    # always plot a priority if present
                    elif len(present_priorities) == 1:
                        df_entry = present_priorities + ['Multiple']
                        sorted_df_entry = cls._sort_list_by_list(df_entry, value_order)
                        parsed_data.loc[category, sample] = tuple(sorted_df_entry + present_borders)

                    # plot two priorities if present, ignoring others
                    elif len(present_priorities) == 2:
                        df_entry = cls._sort_list_by_list(present_priorities, value_order)
                        parsed_data.loc[category, sample] = tuple(df_entry + present_borders)

        return parsed_data

    def _check_samples(self, samples):
        '''Checks that samples are a subset of samples
        currently associated with the CoMut object.

        Params:
        -------
        samples: list-like
            A list of sample names.'''

        if not set(samples).issubset(set(self.samples)):
            extra = set(samples) - set(self.samples)
            raise ValueError('Unknown samples {} given. All added samples'
                             ' must be a subset of either first samples'
                             ' added or samples specified with'
                             ' comut.samples'.format(extra))

    def add_categorical_data(self, data, name=None, category_order=None,
                             value_order=None, mapping=None, borders=None, priority=None,
                             xtick_style='normal', xtick_fontdict=None, xtick_show=True, xtick_rotation=90,
                             ytick_style='normal', ytick_fontdict=None, ytick_show=True, ytick_rotation=0):

        '''Add categorical data to the CoMut object.

        Params:
        -------
        data: pandas dataframe
            A tidy dataframe containing data. Required columns are
            sample, category, and value. Other columns are ignored.

            Example:
            -------
            sample   | category | value
            ----------------------------
            Sample_1 | TP53     | Missense
            Sample_1 | Gender   | Male

        name: str
            The name of the dataset being added. Used to references axes.

            Example:
            --------
            example_comut = comut.CoMut()
            example_comut.add_categorical_data(data, name = 'Mutation type')

        category_order: list-like
            Order of category to plot, from top to bottom. Only these
            categories are plotted.

            Example:
            --------
            example_comut = comut.CoMut()
            example_comut.add_categorical_data(data, category_order = ['TP53', 'BRAF'])

        value_order: list-like
            Order of plotting of values in a single patch, from left
            triangle to right triangle.

            Example:
            --------
            value_order = ['Amp', 'Missense']

            If Amp and Missense exist in the same category and sample, Amp
            will be drawn as left triangle, Missense as right.

        mapping: dict
            Mapping of values to patch properties. The dict can either specify
            only the facecolor or other patches properties.

            Note:
            -----
            Three additional values are required to fully specify mapping:

            'Absent', which determines the color for samples without value
            for a name (default white).

            'Multiple', which determines the color for samples with more than
            two values in that category (default brown).

            'Not Available', which determines the patch properties when a sample's
            value is 'Not Available'.

        borders: list-like
            List of values that should be plotted as borders, not patches.

            Example:
            --------
            example_comut = comut.CoMut()
            example_comut.add_categorical_data(data, borders = ['LOH'])

        priority: list-like
            Ordered list of priorities for values. The function will attempt
            to preserve values in this list, subverting the "Multiple"
            assignment.

            Example:
            --------
            example_comut.add_categorical_data(data, priority = ['Amp'])

            If Amp exists alongside two other values, it will be drawn as
            Amp + Multiple (two triangles), instead of Multiple.

        xtick_style: str, default='normal', 'italic', 'oblique'
            Tick style to be used for the x axis ticks (sample names).

        xtick_fontdict: dict, default=None
            Dictionary controlling the appearance of x axis tick labels (sample names).

        xtick_show: bool, default=True
            Set to False to hide completely x axis ticks and labels (sample names).

        xtick_rotation: bool, default=90
            Rotation in degrees of x axis ticks and labels (sample names).

        ytick_style: str, default='normal', 'italic', 'oblique'
            Tick style to be used for the y axis ticks (category names).

        ytick_fontdict: dict, default=None
            Dictionary controlling the appearance of y axis tick labels (category names).

        ytick_show: bool, default=True
            Set to False to hide completely y axis ticks and labels (category names).

        ytick_rotation: bool, default=0
            Rotation in degrees of y axis ticks and labels (category names).

        Returns:
        --------
        None'''

        # check that required columns exist
        req_cols = {'sample', 'category', 'value'}
        if not req_cols.issubset(data.columns):
            missing_cols = req_cols - set(data.columns)
            msg = ', '.join(list(missing_cols))
            raise ValueError('Data missing required columns: {}'.format(msg))

        # check that samples are a subset of current samples.
        samples = list(data['sample'].drop_duplicates())
        if self.samples is None:
            self.samples = samples
        else:
            self._check_samples(samples)

        # set defaults
        if name is None:
            name = len(self._plots)

        if borders is None:
            borders = []

        if priority is None:
            priority = []

        if value_order is None:
            value_order = []

        # default category order to all categories uniquely present in data
        # in the order they appear
        if category_order is None:
            category_order = list(data['category'].drop_duplicates())

        # build default color map, uses vivid
        unique_values = set(data['value'])
        if mapping is None:
            mapping = {}

            # define default borders
            for value in borders:
                mapping[value] = {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1}

            # assign colors to other unique values
            non_border = [val for val in unique_values if val not in borders]
            default_cmap = self._get_default_categorical_cmap(len(non_border))
            for i, value in enumerate(unique_values):
                 mapping[value] = {'facecolor': default_cmap[i]}

            mapping['Absent'] = {'facecolor': 'white'}
            mapping['Multiple'] = {'facecolor': palettable.colorbrewer.qualitative.Set1_7.mpl_colors[6]}
            mapping['Not Available'] = {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1}

        elif isinstance(mapping, dict):

            # copy the user mapping to avoid overwriting their mapping variable
            mapping = mapping.copy()

            # update user color map with reserved values if not present
            if 'Not Available' not in mapping:
                mapping['Not Available'] = {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1}
            if 'Absent' not in mapping:
                mapping['Absent'] = {'facecolor': 'white'}
            if 'Multiple' not in mapping:
                mapping['Multiple'] = {'facecolor': palettable.colorbrewer.qualitative.Set1_7.mpl_colors[6]}

            # check that all alt types present in data are in mapping
            if not unique_values.issubset(mapping.keys()):
                missing_cats = unique_values - set(mapping.keys())
                raise ValueError('Categories present in dataframe {}'
                                 ' are missing from mapping'.format(missing_cats))

            # if passed values aren't kwargs, convert to patches kwargs
            for key, value in mapping.items():
                if not isinstance(value, dict):
                    if key in borders:
                        mapping[key] = {'facecolor': 'none', 'edgecolor': value}
                    else:
                        mapping[key] = {'facecolor': value}

            # check that borders have facecolor - None
            for border in borders:
                if mapping[border]['facecolor'] != 'none':
                    raise ValueError('Border category {} must have facecolor'
                                     ' = \'none\''.format(border))

        else:
            raise ValueError('Invalid mapping. Mapping must be a dict.')

        # parse data into dataframe of tuples as required for plotting
        parsed_data = self._parse_categorical_data(data, category_order, self.samples,
                                                   value_order, priority, borders)

        # store plot data
        plot_data = {'data': parsed_data, 'patches_options': mapping,
                     'xtick_style': xtick_style, 'xtick_fontdict': xtick_fontdict, 'xtick_show': xtick_show,
                     'xtick_rotation': xtick_rotation,
                     'ytick_style': ytick_style, 'ytick_fontdict': ytick_fontdict, 'ytick_show': ytick_show,
                     'ytick_rotation': ytick_rotation,
                     'borders': borders, 'type': 'categorical'}

        self._plots[name] = plot_data
        return None

    def add_scatter_data(self, data, paired_name=None, name=None, mapping=None, marker="*", markersize=10):
        '''Add a sample level symbol data to the CoMut object

        Params:
        -----------
        data: pandas dataframe
            A tidy dataframe containing data. Required columns are
            sample, category, and value.

        paired_name: str or int
            Name of plot on which the symbols will be placed. Must reference
            a dataset already added to the CoMut object.

        name: str
            The name of the dataset being added. Used to references axes.
            defaults to the integer index of the plot being added.

        mapping: dict
            A mapping of column to colors. Dictionary should map values of column `value`
            to a tuple of colors (str). First and second colors are marker main and alternative colors.

        marker: str, default="*"
            Marker for the scatter plot.

        markersize: float, default=10
            Marker size.

        Returns:
        --------
        None'''
        # check that required columns exist
        req_cols = {'sample', 'category', 'value'}
        if not req_cols.issubset(data.columns):
            missing_cols = req_cols - set(data.columns)
            msg = ', '.join(list(missing_cols))
            raise ValueError('Data missing required columns: {}'.format(msg))

        # check that samples are a subset of current samples.
        samples = list(data['sample'].drop_duplicates())
        if self.samples is None:
            self.samples = samples
        else:
            self._check_samples(samples)

        # set defaults
        if name is None:
            name = len(self._plots)

        # build default color map, uses vivid
        unique_values = set(data['value'])
        if mapping is None:
            mapping = {}

            # define default colors
            default_cmap = self._get_default_categorical_cmap(len(unique_values))
            for i, value in enumerate(unique_values):
                mapping[value] = (default_cmap[i], default_cmap[i])

        elif isinstance(mapping, dict):
            # check that all alt types present in data are in mapping
            if not unique_values.issubset(mapping.keys()):
                missing_cats = unique_values - set(mapping.keys())
                raise ValueError('Categories present in dataframe {}'
                                 ' are missing from mapping'.format(missing_cats))
        else:
            raise ValueError('Invalid mapping. Mapping must be a dict.')


        # convert sample to an index
        data_paired = self._plots[paired_name]["data"]
        data_paired_X = pd.DataFrame({"sample": data_paired.columns, "X": np.arange(0, data_paired.shape[1])})
        data_paired_Y = pd.DataFrame({"category": data_paired.index, "Y": np.arange(0, data_paired.shape[0])})
        data = data.merge(data_paired_X, how="left", on="sample")
        data = data.merge(data_paired_Y, how="left", on="category")

        # store plot data
        plot_data = {'data': data, 'mapping': mapping, 'marker': marker, 'markersize': markersize, 'type': 'scatter'}

        self._over_plots[paired_name][name] = plot_data
        return None


    def add_continuous_data(self, data, mapping='binary',
                             xtick_style='normal', xtick_fontdict=None, xtick_show=True, xtick_rotation=90,
                             ytick_style='normal', ytick_fontdict=None, ytick_show=True, ytick_rotation=0,
                            value_range=None, cat_mapping=None, name=None):
        '''Add a sample level continuous data to the CoMut object

        Params:
        -----------
        data: pandas dataframe
            A tidy dataframe containing data. Required columns are
            sample, category, and value. Other columns are ignored.
            Currently, only one category is allowed.

        mapping: str, colors.LinearSegmentedColormap, default 'binary'
            A mapping of continuous value to color. Can be defined as
            matplotlib colormap (str) or a custom LinearSegmentedColormap
            Samples with missing information are colored according to 'Absent'.

        xtick_style: str, default='normal', 'italic', 'oblique'
            Tick style to be used for the x axis ticks (sample names).

        xtick_fontdict: dict, default=None
            Dictionary controlling the appearance of x axis tick labels (sample names).

        xtick_show: bool, default=True
            Set to False to hide completely x axis ticks and labels (sample names).

        xtick_rotation: bool, default=90
            Rotation in degrees of x axis ticks and labels (sample names).

        ytick_style: str, default='normal', 'italic', 'oblique'
            Tick style to be used for the y axis ticks (category names).

        ytick_fontdict: dict, default=None
            Dictionary controlling the appearance of y axis tick labels (category names).

        ytick_show: bool, default=True
            Set to False to hide completely y axis ticks and labels (category names).

        ytick_rotation: bool, default=0
            Rotation in degrees of y axis ticks and labels (category names).

        value_range: tuple or list
            min and max value of the data. Data will be normalized using
            this range to fit (0, 1). Defaults to the range of the data.

        cat_mapping: dict
            Mapping from a discrete category to patch color. Primarily used
            to override defaults for 'Absent' and 'Not Available' but can
            be used to mix categorical and continuous values in the same data.

        name: str
            The name of the dataset being added. Used to references axes.
            defaults to the integer index of the plot being added.

        tick_style: str, default='normal', 'italic', 'oblique'
            Tick style to be used for the y axis ticks (category names).

        Returns:
        --------
        None'''

        # check that required columns exist
        req_cols = {'sample', 'category', 'value'}
        if not req_cols.issubset(data.columns):
            missing_cols = req_cols - set(data.columns)
            msg = ', '.join(list(missing_cols))
            raise ValueError('Data missing required columns: {}'.format(msg))

        # check that samples are a subset of object samples.
        samples = list(data['sample'].drop_duplicates())
        if self.samples is None:
            self.samples = samples
        else:
            self._check_samples(samples)

        # check that only one category is in the dataframe
        if len(set(data['category'])) > 1:
            raise ValueError('Only one category is allowed for continuous data')

        # make default name
        if name is None:
            name = len(self._plots)

        if value_range is None:
            data_max = pd.to_numeric(data['value'], 'coerce').max()
            data_min = pd.to_numeric(data['value'], 'coerce').min()
        else:
            data_min, data_max = value_range

        # make default categorical mapping
        if cat_mapping is None:
            cat_mapping = {'Absent': {'facecolor': 'white'},
                           'Not Available': {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1}}
        else:
            # update absent and not available
            cat_mapping = cat_mapping.copy()
            if 'Absent' not in cat_mapping:
                cat_mapping['Absent'] = {'facecolor': 'white'}
            if 'Not Available' not in cat_mapping:
                cat_mapping['Not Available'] = {'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 1}

        # if values in cat_mapping aren't kwargs, convert to patches kwargs
        for key, value in cat_mapping.items():
            if not isinstance(value, dict):
                mapping[key] = {'facecolor': value}

        def normalize(x):
            if isinstance(x, (int, float)):
                return (x - data_min)/data_max
            else:
                return x

        # normalize data to range
        norm_data = data.copy()
        norm_data.loc[:, 'value'] = data.loc[:, 'value'].apply(normalize)
        if isinstance(mapping, str):
            mapping = plt.get_cmap(mapping)

        elif not isinstance(mapping, colors.LinearSegmentedColormap):
            raise ValueError('Invalid color map for continuous data. Valid'
                             ' types are colormap str or LinearSegmentedColormap')

        # build color map
        dict_mapping = {}
        for value in norm_data.loc[:, 'value']:
            if isinstance(value, (int, float)):
                dict_mapping[value] = {'facecolor': mapping(value)}

        # update continuous mapping with categorical mapping
        dict_mapping.update(cat_mapping)

        # data is now essentially categorical, so use that to parse data
        category_order = list(norm_data['category'].drop_duplicates())
        parsed_data = self._parse_categorical_data(data=norm_data, category_order=category_order,
                                             sample_order=self.samples, value_order=[], priority=[], borders=[])

        # store plot data
        plot_data = {'data': parsed_data, 'patches_options': dict_mapping,
                     'xtick_style': xtick_style, 'xtick_fontdict': xtick_fontdict, 'xtick_show': xtick_show,
                     'xtick_rotation': xtick_rotation,
                     'ytick_style': ytick_style, 'ytick_fontdict': ytick_fontdict, 'ytick_show': ytick_show,
                     'ytick_rotation': ytick_rotation,
                     'type': 'continuous', 'range': value_range, 'colorbar': mapping}

        self._plots[name] = plot_data
        return None

    def add_bar_data(self, data, name=None, stacked=False, mapping=None, ytick_fontdict=None, ytick_show=True,
                     yaxis_show=True, ylabel='', ylabel_fontsize=None, ylabel_fontweight=None, ylabel_rotation=None,
                     bar_kwargs=None):
        '''Add a bar plot to the CoMut object

        Params:
        -----------
        data: pandas dataframe
            Dataframe containing data for samples. The first column must be
            sample, and other columns should be values for the bar plot.

        name: str
            The name of the dataset being added. Used to references axes.
            Defaults to the integer index of the plot being added.

        stacked: bool, default=False
            Whether the bar graph should be stacked.

        mapping: dict
            A mapping of column to color. Dictionary should map column name
            to color (str) or to plot kwargs.

        ytick_fontdict: dict, default=None
            Dictionary controlling the appearance of y axis tick labels.

        ytick_show: bool, default=True
            Set to False to hide completely y axis ticks and labels.

        yaxis_show: bool, default=True
            Set to False to hide the y-axis line.

        ylabel: str, default ''
            The label for the y axis.

        ylabel_fontsize: float, default=None
            The fontsize of the label for the y axis.

        ylabel_fontweight: str, default=None
            The fontweight of the label for the y axis.

        ylabel_rotation: float, default=None
            The rotation of the label for the y axis.

        bar_kwargs: dict
            dict of kwargs to be passed to plt.bar

        Returns:
        --------
        None'''

        # check formatting
        if data.columns[0] != 'sample':
            raise ValueError('First column in dataframe must be sample')

        # make defaults
        if name is None:
            name = len(self._plots)

        if bar_kwargs is None:
            bar_kwargs = {}

        # convert sample to an index
        bar_df_indexed = data.set_index('sample', drop=True)

        # check that samples are a subset of object samples.
        samples = list(bar_df_indexed.index)
        if self.samples is None:
            self.samples = samples
        else:
            self._check_samples(samples)

            # add missing samples and assign 0 value for all columns
            missing_samples = list(set(self.samples) - set(samples))
            bar_df_indexed = bar_df_indexed.reindex(self.samples)
            bar_df_indexed.loc[missing_samples, :] = 0

        # make default mapping
        if mapping is None:
            num_cats = len(bar_df_indexed.columns)
            default_cmap = self._get_default_categorical_cmap(num_cats)
            mapping = {column: default_cmap[i]
                       for i, column in enumerate(bar_df_indexed.columns)}

        # store plot data
        plot_data = {'data': bar_df_indexed, 'bar_options': mapping, 'type': 'bar',
                     'stacked': stacked, 'ytick_fontdict': ytick_fontdict, 'ytick_show': ytick_show,
                     'ylabel': ylabel, 'ylabel_fontsize': ylabel_fontsize, 'ylabel_fontweight': ylabel_fontweight,
                     'ylabel_rotation': ylabel_rotation, 'yaxis_show': yaxis_show, 'bar_kwargs': bar_kwargs}

        self._plots[name] = plot_data
        return None

    def add_sample_indicators(self, data, name=None, xtick_style='normal', xtick_fontdict=None,
                              xtick_show=True, xtick_rotation=90, plot_kwargs=None):
        '''Add a line plot that indicates samples that share a characteristic

        Params:
        -----------
        data: pandas dataframe
            A tidy dataframe that assigns individual samples to groups.
            Required columns are 'sample' and 'group'. Other columns are
            ignored.

        name: str
            The name of the dataset being added. Used to references axes
            Defaults to the integer index of the plot being added.

        xtick_style: str, default='normal', 'italic', 'oblique'
            Tick style to be used for the x axis ticks (sample names).

        xtick_fontdict: dict, default=None
            Dictionary controlling the appearance of x axis tick labels (sample names).

        xtick_show: bool, default=True
            Set to False to hide completely x axis ticks and labels (sample names).

        xtick_rotation: bool, default=90
            Rotation in degrees of x axis ticks and labels (sample names).

        plot_kwargs: dict
            dict of kwargs to be passed to plt.plot during plotting. Defaults
            to {'color': 'black', 'marker': 'o', markersize': 3}

        Returns:
        --------
        None'''

        # check for required columns
        req_cols = {'sample', 'group'}
        if not req_cols.issubset(data.columns):
            missing_cols = req_cols - set(data.columns)
            msg = ', '.join(list(missing_cols))
            raise ValueError('Data missing required columns: {}'.format(msg))

        # make defaults
        if name is None:
            name = len(self._plots)

        if plot_kwargs is None:
            plot_kwargs = {'color': 'black', 'marker': 'o', 'markersize': 3}

        # convert sample to an index
        data_indexed = data.set_index('sample', drop=True)

        # check that samples are a subset of current samples
        samples = list(data_indexed.index)
        if self.samples is None:
            self.samples = samples
        else:
            self._check_samples(samples)

            # add missing samples and assign them NaN. They will be skipped.
            missing_samples = list(set(self.samples) - set(samples))

            # Reorders - by default uses new samples = NaN
            data_indexed = data_indexed.reindex(self.samples)

        # connected samples must be adjacent. Throw an error otherwise.
        seen_groups = set()
        prev_group = None
        for assignment in data_indexed['group']:
            if assignment in seen_groups and not np.isnan(assignment):
                raise ValueError('Samples that share a group must be adjacent'
                                 ' in CoMut sample ordering.')
            elif assignment != prev_group:
                seen_groups.add(prev_group)
                prev_group = assignment

        plot_data = {'data': data_indexed, 'plot_options': plot_kwargs,
                     'xtick_style': xtick_style, 'xtick_fontdict': xtick_fontdict, 'xtick_show': xtick_show,
                     'xtick_rotation': xtick_rotation, 'type': 'indicator'}

        self._plots[name] = plot_data
        return None

    def _plot_patch_data(self, ax, data, name, mapping, borders,
                         ytick_style='normal', ytick_fontdict=None, ytick_show=True, ytick_rotation=0,
                         x_padding=0, y_padding=0, tri_padding=0):
        '''Plot data represented as patches on CoMut plot

        Params:
        -----------
        ax: axis object
            Axis object on which to draw the graph.

        data: pandas dataframe
            Parsed dataframe from _parse_categorical_data

        name: str
            Name of the plot to store in axes dictionary.

        mapping: dict
            mapping from add_categorical_data

        borders: list-like
            borders from add_categorical_data

        ytick_style: str, default='normal', 'italic', 'oblique'
            Tick style to be used for the y axis ticks (category names).

        ytick_fontdict: dict, default=None
            Dictionary controlling the appearance of y axis tick labels (category names).

        ytick_show: bool, default=True
            Set to False to hide completely y axis ticks and labels (category names).

        ytick_rotation: bool, default=0
            Rotation in degrees of y axis ticks and labels (category names).

        x_padding: float, default=0
            x_padding from plot_comut

        y_padding: float, default=0
            y_padding from plot_comut

        tri_padding: float, default=0
            tri_padding from plot_comut


        Returns:
        -------
        ax: axis object
            Axis object on which the plot is drawn.'''

        # precalculate height and width of patches
        height, width = 1 - 2*y_padding, 1 - 2*x_padding

        # store unique labels
        unique_labels = set()

        # plot data from bottom to top, left to right of CoMut
        for i in range(len(data.index)):
            for j in range(len(data.columns)):

                # calculate loc of lower left corner of patch
                x_base, y_base = j + x_padding, i + y_padding

                cell_tuple = tuple(data.iloc[i, j])

                # remove box borders from cell tuple
                box_borders = [value for value in cell_tuple
                               if value in borders]

                # determine the values that are not borders
                cell_tuple = [value for value in cell_tuple
                              if value not in borders]

                # determine number of values
                num_values = len(cell_tuple)

                # plot Not Available if present
                if 'Not Available' in cell_tuple:
                    if len(cell_tuple) > 1:
                        raise ValueError('Not Available must be a value by itself')

                    # otherwise plot the Not Available patch. label = '' subverts legend
                    patch_options = mapping['Not Available']
                    rect = patches.Rectangle((x_base, y_base), width, height, **patch_options,
                                             label='')
                    ax.add_patch(rect)

                    # prevent the border from exceeding the bounds of the patch
                    rect.set_clip_path(rect)

                    # plot the slashed line. This code is heuristic and does
                    # not currently scale well.
                    ax.plot([x_base + x_padding/2, x_base + width - x_padding/2],
                            [y_base + y_padding/2, y_base + height - y_padding/2],
                             color=patch_options['edgecolor'], linewidth=0.5,
                             solid_capstyle='round')

                    # go to next patch
                    continue

                # use rectangles to draw single boxes
                if num_values != 2:

                    # handle Absent and Multiple
                    if num_values == 0:
                        label = 'Absent'
                        patch_options = mapping['Absent']
                    elif num_values > 2:
                        label = 'Multiple'
                        patch_options = mapping['Multiple']

                    # extract color for single patch based on value
                    elif num_values == 1:
                        value = cell_tuple[0]
                        label = value
                        patch_options = mapping[value]

                    # create rectangle and add to plot. Add label if it
                    # doesn't already exist in legend
                    plot_label = label if label not in unique_labels else None
                    unique_labels.add(label)
                    rect = patches.Rectangle((x_base, y_base),
                                             width, height,
                                             **patch_options, label=plot_label)
                    ax.add_patch(rect)

                # if two alterations, build using two triangles
                else:
                    alt_1, alt_2 = cell_tuple

                    # determine if labels are unique and add if so
                    alt_1_label = alt_1 if alt_1 not in unique_labels else None
                    unique_labels.add(alt_1)
                    alt_2_label = alt_2 if alt_2 not in unique_labels else None
                    unique_labels.add(alt_2)

                    # extract color options for triangles
                    patch_options_1 = mapping[alt_1]
                    patch_options_2 = mapping[alt_2]

                    # build triangles with triangle padding
                    tri_1, tri_2 = self._get_triangles(x_base, y_base, tri_padding,
                                                 height, width)

                    tri_1_patch = patches.Polygon(tri_1, label=alt_1_label, **patch_options_1)
                    tri_2_patch = patches.Polygon(tri_2, label=alt_2_label, **patch_options_2)

                    ax.add_patch(tri_1_patch)
                    ax.add_patch(tri_2_patch)

                # Once boxes have been plotted, plot border
                for value in box_borders:
                    border_options = mapping[value]
                    rect = patches.Rectangle((x_base, y_base),
                                             width, height,
                                             **border_options, label=value)
                    ax.add_patch(rect)
                    rect.set_clip_path(rect)

        # x and y limits
        ax.set_ylim([0, len(data.index) + y_padding])
        ax.set_xlim([0, len(data.columns) + x_padding])

        # customize ytick labels
        if ytick_show:
            ax.set_yticks(np.arange(0.5, len(data.index) + 0.5))
            ax.set_yticklabels(data.index, fontdict=ytick_fontdict, style=ytick_style, rotation=ytick_rotation)
        else:
            ax.set_yticklabels([])

        # delete tick marks and make x axis invisible
        ax.get_xaxis().set_visible(False)
        ax.tick_params(
            axis='both',       # changes apply to both axes
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            length=0)          # remove ticks

        # remove spines
        for loc in ['top', 'right', 'bottom', 'left']:
            ax.spines[loc].set_visible(False)

        self.axes[name] = ax
        return ax

    def _plot_scatter_data(self, ax, data, name, mapping, marker, markersize):
        for value, (color, coloralt) in mapping.items():
            mask_data = data["value"]==value
            ax.plot(data.loc[mask_data]["X"]+0.5, data.loc[mask_data]["Y"]+0.5, marker=marker, markersize=markersize,
                    color=color, markerfacecoloralt=coloralt, linestyle='', markeredgecolor='None', fillstyle='left',
                    markeredgewidth=0)

        return ax

    def _plot_bar_data(self, ax, data, name, mapping, stacked, ytick_fontdict, ytick_show, yaxis_show, ylabel,
                       ylabel_fontsize, ylabel_fontweight, ylabel_rotation, bar_kwargs):
        '''Plot bar plot on CoMut plot

        Params:
        -----------
        ax: axis object
            axis object on which to draw the graph.

        data: pandas Dataframe
            Dataframe from add_bar_data

        name: str
            name from add_bar_data

        mapping: dict
            mapping from add_bar_data

        stacked: bool
            stacked from add_bar_data

        ytick_fontdict: dict
            ytick_fontdict from add_bar_data

        ytick_show: bool
            ytick_show from add_bar_data

        yaxis_show: bool
            yaxis_show from add_bar_data

        ylabel: str
            ylabel from add_bar_data

        ylabel_fontsize: float
            ylabel_fontsize from add_bar_data

        ylabel_fontweight: str
            ylabel_fontweight from add_bar_data

        ylabel_rotation: float
            ylabel_rotation from add_bar_data

        bar_kwargs: dict
            bar_kwargs from add_bar_data

        Returns:
        -------
        ax: axis object
            Axis object on which the plot is drawn'''

        # define x range
        x_range = np.arange(0.5, len(data.index))

        # if stacked, calculate cumulative height of bars
        if stacked:
            cum_bar_df = np.cumsum(data, axis=1)

            # for each bar, calculate bottom and top of bar and plot it
            for i in range(len(cum_bar_df.columns)):
                column = cum_bar_df.columns[i]
                color = mapping[column]
                if i == 0:
                    bottom = None
                    bar_data = cum_bar_df.loc[:, column]
                else:
                    # calculate distance between previous and current column
                    prev_column = cum_bar_df.columns[i-1]
                    bar_data = cum_bar_df.loc[:, column] - cum_bar_df.loc[:, prev_column]

                    # the previous column defines the bottom of the bars
                    bottom = cum_bar_df.loc[:, prev_column]

                # plot bar data
                ax.bar(x_range, bar_data, align='center', color=color,
                       bottom=bottom, label=column, **bar_kwargs)

            y_max = cum_bar_df.iloc[:,-1].max()

        # plot unstacked bar. Label is '' to subvert legend.
        else:
            color = mapping[data.columns[0]]
            ax.bar(x_range, data.iloc[:, 0],
                   align='center', color=color, label='', **bar_kwargs)

            y_max = data.iloc[:,0].max()


        # customize ytick labels
        if ytick_show:
            ax.set_yticks([0, y_max])
            ax.set_yticklabels([0, y_max], fontdict=ytick_fontdict, style="normal", rotation=0)
        else:
            ax.set_yticklabels([])

        # set the ylabel
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontweight=ylabel_fontweight, rotation=ylabel_rotation)

        # make x axis invisible
        ax.get_xaxis().set_visible(False)

        if yaxis_show:
            # despine all axes except left
            for loc in ['top', 'right', 'bottom']:
                ax.spines[loc].set_visible(False)
        else:
            # despine all axes except left
            for loc in ['top', 'right', 'bottom', 'left']:
                ax.spines[loc].set_visible(False)

        self.axes[name] = ax
        return ax

    def add_side_bar_data(self, data, paired_name, name=None, position='right', mapping=None, stacked=False,
                          xtick_fontdict=None, xtick_show=True, xaxis_show=True, xlabel='', xlabel_fontsize=None,
                          xlabel_fontweight=None, xlabel_rotation=None, gap_between_groups=0.2, gap_within_groups=0.05,
                          bar_kwargs=None):
        '''Add a side bar plot to the CoMut object

        Params:
        -----------
        data: pandas dataframe
            Dataframe containing data for categories in paired plot. The first
            column must be category, and other columns should be values for the
            bar plot.

        paired_name: str or int
            Name of plot on which the bar plot will be placed. Must reference
            a dataset already added to the CoMut object.

        name: str
            The name of the dataset being added. Used to references axes.
            defaults to the integer index of the plot being added.

        position: str, 'left' or 'right', default 'right'
            Where the bar plot should be graphed (left or right of paired name
            plot).

        stacked: bool, default=False
            Whether the bar graph should be stacked.

        mapping: dict
            A mapping of column to color. Dictionary should map column name
            to color (str) or to plot kwargs.

        xtick_fontdict: dict, default=None
            Dictionary controlling the appearance of x axis tick labels.

        xtick_show: bool, default=True
            Set to False to hide completely x axis ticks and labels.

        xaxis_show: bool, default=True
            Set to False to hide the x-axis line.

        xlabel: str, default ''
            The label for the x axis.

        xlabel_fontsize: float, default=None
            The fontsize of the label for the x axis.

        xlabel_fontweight: str, default=None
            The fontweight of the label for the x axis.

        xlabel_rotation: float, default=None
            The rotation of the label for the x axis.

        gap_between_groups: float
            gap in [0,1] between groups of bars

        gap_within_groups: float
            gap in [0,1] between bars from the same group

        bar_kwargs: dict
            kwargs to be passed to plt.barh during the process of plotting.

        Returns:
        --------
        None'''

        # check formatting
        if data.columns[0] != 'category':
            raise ValueError('First column in dataframe must be category')

        # make defaults
        if name is None:
            name = len(self._plots)

        if position not in ['left', 'right']:
            raise ValueError('Position must be left or right')

        if bar_kwargs is None:
            bar_kwargs = {}

        # side plots must be paired with a plot that exists
        if paired_name not in self._plots:
            raise KeyError('Plot {} does not exist. Side plots must be added'
                           'to an already existing plot'.format(paired_name))

        # currently, side plot must be paired with a categorical dataset
        paired_plot = self._plots[paired_name]
        if paired_plot['type'] != 'categorical':
            raise ValueError('Side plots can only be added to categorical data')

        # set index to categories
        data_indexed = data.set_index('category')

        # check that the categories match paired plot's categories
        side_cats = set(data_indexed.index)
        paired_cats = set(paired_plot['data'].index)
        if not side_cats.issubset(paired_cats):
            new_cats = side_cats - paired_cats
            raise ValueError('Categories {} do not exist in paired plot {}. '
                             'Categories in side bar plot must be a subset of'
                             ' those in paired plot.'.format(new_cats, paired_name))

        # add missing categories and assign them a value of 0 for all rows
        missing_categories = paired_cats - side_cats
        data_indexed = data_indexed.reindex(list(paired_plot['data'].index))
        data.loc[list(missing_categories), :] = 0

        # make default mapping
        if mapping is None:
            mapping = {column: palettable.cartocolors.qualitative.Vivid_10.mpl_colors[i]
                       for i, column in enumerate(data_indexed.columns)}

        # store the data
        plot_data = {'data': data_indexed, 'mapping': mapping, 'stacked': stacked, 'position': position,
                     'xtick_fontdict': xtick_fontdict, 'xtick_show': xtick_show,
                     'xlabel': xlabel, 'xlabel_fontsize': xlabel_fontsize, 'xlabel_fontweight': xlabel_fontweight,
                     'xlabel_rotation': xlabel_rotation, 'xaxis_show': xaxis_show,
                     'gap_between_groups': gap_between_groups, 'gap_within_groups': gap_within_groups,
                     'bar_kwargs': bar_kwargs, 'type': 'bar'}

        self._side_plots[paired_name][name] = plot_data
        return None


    def add_side_error_data(self, data, paired_name, name=None, position='right', mapping=None,
                          xtick_fontdict=None, xtick_show=True, xaxis_show=True, xlabel='', xlabel_fontsize=None,
                          xlabel_fontweight=None, xlabel_rotation=None):
        '''Add a side bar plot to the CoMut object

        Params:
        -----------
        data: pandas dataframe
            Dataframe containing data for categories in paired plot. The first
            column must be category, and other columns should be values for the
            bar plot.

        paired_name: str or int
            Name of plot on which the bar plot will be placed. Must reference
            a dataset already added to the CoMut object.

        name: str
            The name of the dataset being added. Used to references axes.
            defaults to the integer index of the plot being added.

        position: str, 'left' or 'right', default 'right'
            Where the bar plot should be graphed (left or right of paired name
            plot).

        mapping: dict
            A mapping of subcategory to graphical parameters. Dictionary should map column name
            to list of graphical attributes in the following order:
                - 'pos' deviation (usually between -0.5 and 0.5 from the ith position)
                - 'col' color of the central marker
                - 'ecol' color of the edge line
                - 'els' line style of the edge line
                - 'elw' line width of the edge line
                - 'fmt' format of the central marker
                - 'ms' marker size

        xtick_fontdict: dict, default=None
            Dictionary controlling the appearance of x axis tick labels.

        xtick_show: bool, default=True
            Set to False to hide completely x axis ticks and labels.

        xaxis_show: bool, default=True
            Set to False to hide the x-axis line.

        xlabel: str, default ''
            The label for the x axis.

        xlabel_fontsize: float, default=None
            The fontsize of the label for the x axis.

        xlabel_fontweight: str, default=None
            The fontweight of the label for the x axis.

        xlabel_rotation: float, default=None
            The rotation of the label for the x axis.

        Returns:
        --------
        None'''

        # check formatting
        if data.columns[0] != 'category':
            raise ValueError('First column in dataframe must be category')


        if data.columns[1] != 'subcategory':
            raise ValueError('Second column in dataframe must be subcategory')

        # make defaults
        if name is None:
            name = len(self._plots)

        if position not in ['left', 'right']:
            raise ValueError('Position must be left or right')

        # side plots must be paired with a plot that exists
        if paired_name not in self._plots:
            raise KeyError('Plot {} does not exist. Side plots must be added'
                           'to an already existing plot'.format(paired_name))

        # currently, side plot must be paired with a categorical dataset
        paired_plot = self._plots[paired_name]
        if paired_plot['type'] != 'categorical':
            raise ValueError('Side plots can only be added to categorical data')

        # check that the categories match paired plot's categories
        side_cats = set(data["category"])
        paired_cats = set(paired_plot['data'].index)
        if not side_cats.issubset(paired_cats):
            new_cats = side_cats - paired_cats
            raise ValueError('Categories {} do not exist in paired plot {}. '
                             'Categories in side bar plot must be a subset of'
                             ' those in paired plot.'.format(new_cats, paired_name))

        # make default mapping
        if mapping is None:
            subcategories = list(data["subcategory"].unique())
            n_subcategories = len(subcategories)
            vivid_10 =  palettable.cartocolors.qualitative.Vivid_10.mpl_colors
            mapping = {}

            for i, subcategory in enumerate(subcategories):
                pos = -0.3 + 0.6*i/(n_subcategories-1)
                col = vivid_10[i]
                ecol = col
                els = "-"
                elw = 1
                fmt = "o"
                ms = 6
                mapping[subcategory] = [pos, col, ecol, els, elw, fmt, ms]


        # add Y for positioning bars
        data_paired = self._plots[paired_name]["data"]
        data_paired_Y = pd.DataFrame({"category": data_paired.index, "Y": np.arange(0, data_paired.shape[0])})
        data = data.merge(data_paired_Y, how="left", on="category")

        # store the data
        plot_data = {'data': data, 'mapping': mapping, 'position': position,
                     'xtick_fontdict': xtick_fontdict, 'xtick_show': xtick_show,
                     'xlabel': xlabel, 'xlabel_fontsize': xlabel_fontsize, 'xlabel_fontweight': xlabel_fontweight,
                     'xlabel_rotation': xlabel_rotation, 'xaxis_show': xaxis_show,
                     'type': 'error'}

        self._side_plots[paired_name][name] = plot_data
        return None

    def _plot_indicator_data(self, ax, data, name, plot_kwargs):
        '''Plot data that connects samples with similar characteristics.

        Params:
        -----------
        ax: axis object
            axis object on which to draw the graph.

        data: pandas dataframe
            data from add_sample_indicators

        name: str
            name from add_sample_indicators

        plot_kwargs: dict
            plot_kwargs from add_sample_indicators

        Returns:
        --------
        ax: axis object
            Axis object on which the plot is drawn.'''

        # loop through group assignments
        for i, group in enumerate(set(data['group'])):

            # ignore missing samples
            if np.isnan(group):
                continue

            # plot the first with a label so legend can extract it later
            label = name if i == 0 else None

            # extract x coordinates of group members.
            x_vals = np.where(data['group'] == group)[0]

            # plot line plot connecting samples in group
            ax.plot(x_vals + 0.5, [0.5]*len(x_vals), label=label, **plot_kwargs)

        # make axes invisible and despine
        for loc in ['top', 'right', 'bottom', 'left']:
            ax.spines[loc].set_visible(False)

        # remove axes
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        self.axes[name] = ax
        return self

    def _plot_side_bar_data(self, ax, name, data, mapping, position, stacked, xtick_fontdict, xtick_show, xaxis_show,
                            xlabel, xlabel_fontsize, xlabel_fontweight, xlabel_rotation, y_padding,
                            gap_between_groups, gap_within_groups, bar_kwargs):
        '''Plot side bar plot on CoMut plot

        Params:
        -----------
        ax: axis object
            axis object on which to draw the graph.

        data: pandas Dataframe
            data from add_side_bar_data

        name: str
            name from add_side_bar_data

        mapping: dict
            mapping from add_side_bar_data

        position: str, left or right
            position from add_side_bar_data

        stacked: bool
            stacked from add_side_bar_data

        xtick_fontdict: dict
            xtick_fontdict from add_side_bar_data

        xtick_show: bool
            xtick_show from add_side_bar_data

        xaxis_show: bool
            xaxis_show from add_side_bar_data

        xlabel: str
            xlabel from add_side_bar_data

        xlabel_fontsize: float
            xlabel_fontsize from add_side_bar_data

        xlabel_fontweight: str
            xlabel_fontweight from add_side_bar_data

        xlabel_rotation: float
            xlabel_rotation from add_side_bar_data

        y_padding: float
            y_padding from plot_comut

        gap_between_groups: float
            gap_between_groups from add_side_bar_data

        gap_within_groups: float
            gap_within_groups from add_side_bar_data

        bar_kwargs: dict
            bar_kwargs from add_side_bar_data

        Returns:
        -------
        ax: axis object
            The axis object on which the plot is drawn.'''

        # define y range, since the plot is rotated
        y_range = np.arange(0.5, len(data.index))

        # if stacked, calculate cumulative height of bars
        if stacked:
            # if height not in bar_kwargs, set it
            if 'height' not in bar_kwargs:
                bar_kwargs['height'] = 1 - 2*y_padding

            cum_bar_df = np.cumsum(data, axis=1)

            # for each bar, calculate bottom and top of bar and plot
            for i in range(len(cum_bar_df.columns)):
                column = cum_bar_df.columns[i]
                color = mapping[column]
                if i == 0:
                    left = None
                    bar_data = cum_bar_df.loc[:, column]
                else:
                    # calculate distance between previous and current column
                    prev_column = cum_bar_df.columns[i-1]
                    bar_data = cum_bar_df.loc[:, column] - cum_bar_df.loc[:, prev_column]

                    # previous column defines the "bottom" of the bars
                    left = cum_bar_df.loc[:, prev_column]

                ax.barh(y_range, bar_data, align='center', color=color,
                        left=left, label=column, **bar_kwargs)

            x_max = cum_bar_df.iloc[:,-1].max()

        # plot unstacked bar
        else:
            n_bars = data.shape[1]
            height = (1-gap_between_groups-(n_bars-1)*gap_within_groups)/n_bars

            for i, col in enumerate(data.columns):
                x_range = data[col]
                color =  mapping[col]
                y_range = -0.5 + gap_between_groups/2 + (i+0.5) * height + i * gap_within_groups + \
                        np.arange(0.5, len(x_range))
                ax.barh(y_range, x_range, align='center', color=color, height=height, **bar_kwargs)

            x_max = data.max().max()

        # reverse x axis if position is to the left
        if position == 'left':
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[::-1])

        for loc in ['top', 'right', 'left']:
            ax.spines[loc].set_visible(False)

        # set xlabel
        ax.set_xlabel(xlabel)

        # customize xtick labels
        if xtick_show:
            ax.set_xticks([0, x_max])
            ax.set_xticklabels([0, x_max], fontdict=xtick_fontdict, style="normal", rotation=0)
        else:
            ax.set_xticklabels([])

        # set the xlabel
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontweight=xlabel_fontweight, rotation=xlabel_rotation)

        # make y axis invisible
        ax.get_yaxis().set_visible(False)

        if xaxis_show:
            # despine all axes except bottom
            for loc in ['top', 'right', 'left']:
                ax.spines[loc].set_visible(False)
        else:
            # despine all axes except left
            for loc in ['top', 'right', 'bottom', 'left']:
                ax.spines[loc].set_visible(False)

        self.axes[name] = ax
        return ax


    def _plot_side_error_data(self, ax, name, data, mapping, position, xtick_fontdict, xtick_show, xaxis_show,
                              xlabel, xlabel_fontsize, xlabel_fontweight, xlabel_rotation, y_padding):
        '''Plot side bar plot on CoMut plot

        Params:
        -----------
        ax: axis object
            axis object on which to draw the graph.

        data: pandas Dataframe
            data from add_side_error_data

        name: str
            name from add_side_error_data

        mapping: dict
            mapping from add_side_error_data

        position: str, left or right
            position from add_side_error_data

        xtick_fontdict: dict
            xtick_fontdict from add_side_error_data

        xtick_show: bool
            xtick_show from add_side_error_data

        xaxis_show: bool
            xaxis_show from add_side_error_data

        xlabel: str
            xlabel from add_side_error_data

        xlabel_fontsize: float
            xlabel_fontsize from add_side_error_data

        xlabel_fontweight: str
            xlabel_fontweight from add_side_error_data

        xlabel_rotation: float
            xlabel_rotation from add_side_error_data

        y_padding: float
            y_padding from plot_comut

        Returns:
        -------
        ax: axis object
            The axis object on which the plot is drawn.'''

        for subcategory in data["subcategory"].unique():
            mask_sub = data["subcategory"]==subcategory
            pos = mapping[subcategory][0]
            col = mapping[subcategory][1]
            ecol = mapping[subcategory][2]
            els = mapping[subcategory][3]
            elw = mapping[subcategory][4]
            fmt = mapping[subcategory][5]
            ms = mapping[subcategory][6]

            eb = ax.errorbar(x=data.loc[mask_sub]["value"], y=data.loc[mask_sub]["Y"] + 0.5 + pos,
                             xerr=np.abs(data.loc[mask_sub][['low', 'high']].values.T - data.loc[mask_sub]["value"].values),
                             fmt=fmt, color=col, ecolor=ecol, elinewidth=elw, ms=ms, capsize=0)

        # reverse x axis if position is to the left
        if position == 'left':
            ax.set_xlim(xlim[::-1])

        for loc in ['top', 'right', 'left']:
            ax.spines[loc].set_visible(False)

        # set xlabel
        ax.set_xlabel(xlabel)

        # customize xtick labels
        if xtick_show:
            x_max = round(data["high"].max(), 1)
            x_min = round(data["low"].min(), 1)
            if x_min < 1 and x_max > 1:
                xticks = [x_min, 1, x_max]
            elif x_min < 1:
                xticks = [x_min, 1]
            elif x_max > 1:
                xticks = [1, x_max]
            else:
                xticks = [1]

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontdict=xtick_fontdict, style="normal", rotation=0)
        else:
            ax.set_xticklabels([])

        # set the xlabel
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontweight=xlabel_fontweight, rotation=xlabel_rotation)

        # make y axis invisible
        ax.get_yaxis().set_visible(False)

        if xaxis_show:
            # despine all axes except bottom
            for loc in ['top', 'right', 'left']:
                ax.spines[loc].set_visible(False)
        else:
            # despine all axes except left
            for loc in ['top', 'right', 'bottom', 'left']:
                ax.spines[loc].set_visible(False)

        self.axes[name] = ax
        return ax


    def _plot_data_on_axis(self, ax, plot_name, plot_type, plot_params, data, x_padding, y_padding, tri_padding):
        '''Wrapper function for plotting data on an axis

        Params:
        -------
        ax: axis object
            Axis object on which to plot

        plot_name: str
            Name of plot, used to index into plot dictionary associated
            with CoMut object.

        plot_type: str
            Type of plot, among 'categorical', 'continuous', 'scatter' or 'bar'.

        plot_params: dict
            A dictionary with all the parameters required for drawing the plot.

        data: pandas dataframe
            Table containing the data underlying the plot.

        x_padding, y_padding: float
            Padding within patches for categorical data

        tri_padding: float
            Padding between triangles for categorical data

        Returns:
        --------
        ax: The axis object with now plotted_data'''

        # extract relevant plotting params depending on plot type, then plot
        if plot_type == 'categorical' or plot_type == 'continuous':
            mapping = plot_params['patches_options']
            borders = plot_params['borders'] if plot_type == 'categorical' else []
            ytick_style = plot_params['ytick_style']
            ytick_fontdict = plot_params['ytick_fontdict']
            ytick_show = plot_params['ytick_show']
            ytick_rotation = plot_params['ytick_rotation']
            ax = self._plot_patch_data(ax=ax, data=data, name=plot_name, mapping=mapping, borders=borders,
                                       ytick_style=ytick_style, ytick_fontdict=ytick_fontdict, ytick_show=ytick_show,
                                       ytick_rotation=ytick_rotation,
                                       x_padding=x_padding, y_padding=y_padding, tri_padding=tri_padding)

        elif plot_type == "scatter":
            mapping = plot_params['mapping']
            marker = plot_params['marker']
            markersize = plot_params['markersize']
            ax = self._plot_scatter_data(ax=ax, data=data, name=plot_name, mapping=mapping, marker=marker,
                                         markersize=markersize)

        elif plot_type == 'bar':
            mapping = plot_params['bar_options']
            stacked = plot_params['stacked']
            ytick_fontdict = plot_params['ytick_fontdict']
            ytick_show = plot_params['ytick_show']
            yaxis_show = plot_params['yaxis_show']
            ylabel = plot_params['ylabel']
            ylabel_fontsize = plot_params['ylabel_fontsize']
            ylabel_fontweight = plot_params['ylabel_fontweight']
            ylabel_rotation = plot_params['ylabel_rotation']
            bar_kwargs = plot_params['bar_kwargs']

            # set the default width based on padding if not specified for bars
            if 'width' not in bar_kwargs:
                bar_kwargs['width'] = 1 - 2*x_padding
            ax = self._plot_bar_data(ax=ax, data=data, name=plot_name, mapping=mapping, stacked=stacked,
                                     ytick_fontdict=ytick_fontdict, ytick_show=ytick_show, yaxis_show=yaxis_show,
                                     ylabel=ylabel, ylabel_fontsize=ylabel_fontsize,
                                     ylabel_fontweight=ylabel_fontweight, ylabel_rotation=ylabel_rotation,
                                     bar_kwargs=bar_kwargs)

        elif plot_type == 'indicator':
            plot_kwargs = plot_params['plot_options']
            ax = self._plot_indicator_data(ax=ax, data=data, name=plot_name, plot_kwargs=plot_kwargs)

        return ax

    def _get_default_widths_and_comut_loc(self):
        '''Gets default widths from plots present in side_plots,
        as well as the index location of the CoMut in widths.

        Returns:
        --------
        default_widths: list
            Default widths (5 to central CoMut, 1 to side plots).

        comut_idx: int
            Integer index of the CoMut plot in the width list
        '''

        # determine the maximum number of right and left plots
        max_left, max_right = 0, 0
        for side_plots in self._side_plots.values():
            positions = [side_plot['position'] for side_plot in side_plots.values()]

            if positions.count('left') > max_left:
                max_left = positions.count('left')
            if positions.count('right') > max_right:
                max_right = positions.count('right')

        # CoMut gets rel width of 5, other plots get 1
        default_widths = [1]*max_left + [5] + [1]*max_right

        # CoMut is located in between left and right plots
        comut_idx = max_left

        return default_widths, comut_idx

    def _get_default_height(self, name, plot_type):
        '''Returns default height for a plot

        Params:
        -------
        name: str or int
            Name of the plot.

        plot_type: str
            Type of plot, used to set default height.

        Returns:
        --------
        height: float
            Default height for plot type'''

        if plot_type == 'categorical':
            data = self._plots[name]['data']
            height = len(data)

        elif plot_type == 'continuous':
            height = 1

        elif plot_type == 'bar':
            height = 3

        elif plot_type == 'indicator':
            height = 1

        else:
            raise ValueError('Invalid plot type {}'.format(plot_type))

        return height

    def _get_height_spec(self, structure, heights):
        '''Gets the default heights for plots in the CoMut.

        Height values for each plot type default to the following:
            Number of categories for categorical data
            1 for continuous data
            3 for bar plots
            1 for sample indicator

        Params:
        -------
        structure: list of lists
            The structure of the CoMut plot, given as a list of lists
            containing plot names.

        heights: dict
            Dictionary specifying the relative height of certain plots.
            Keys must be plot names, and values must be relative heights.

        Returns:
        --------
        height_structure: list of lists
            Relative heights in the same form as structure

        Example:
        --------
        heights = {'plot1': 3, 'plot2': 5, 'plot3': 7}

        # plot4 is a bar plot
        structure = [['plot1', 'plot2'], ['plot3'], ['plot4']]

        # returns [[3, 5], [7], [3]]
        get_height_structure(heights, structure)'''

        structure_heights = []
        for plot in structure:
            # if one plot in subplot, make it appropriate size
            if len(plot) == 1:
                name = plot[0]

                # get default height if no user height specified
                plot_type = self._plots[name]['type']
                if name in heights:
                    plot_height = heights[name]
                else:
                    plot_height = self._get_default_height(name, plot_type)

                structure_heights.append([plot_height])

            # if more than one, calculate height of all plots in subplot
            else:
                subplot_heights = []
                for name in plot:
                    plot_type = self._plots[name]['type']

                    # get default height if no user height specified
                    if name in heights:
                        plot_height = heights[name]
                    else:
                        plot_height = self._get_default_height(name, plot_type)

                    subplot_heights.append(plot_height)
                structure_heights.append(subplot_heights)

        # structure heights should match the shape of structure
        return structure_heights

    def plot_comut(self, fig=None, spec=None, x_padding=0, y_padding=0,
                   tri_padding=0, heights=None, hspace=0.2, subplot_hspace=None,
                   widths=None, wspace=0.2, shadow_width_left=None,
                   structure=None, figsize=(10,6)):
        '''plot the CoMut object

        Params:
        -----------
        fig: `~.figure.Figure`
            The figure on which to create the CoMut plot. If no fig
            is passed, it will be created.

        spec: gridspec
            The gridspec on which to create the CoMut plot. If no spec
            is passed, one will be created.

        x_padding, y_padding: float, optional (default 0)
            The space between patches in the CoMut plot in the x and y
            direction.

        tri_padding: float
            If there are two values for a sample in a category, the spacing
            between the triangles that represent each value.

        heights: dict
            The relative heights of all the plots. Dict should have keys as
            plot names and values as relative height.

            Height values for each plot type default to the following:
                Number of categories for categorical data
                1 for continuous data
                3 for bar plots
                1 for sample indicator

            Example:
            --------
            heights = {'plot1': 3, 'plot2': 5, 'plot3': 7}
            CoMut.plot_comut(heights=heights)

        hspace: float, default 0.2
            The distance between different plots in the CoMut plot.

        widths: list-like
            The relative widths of plots in the x direction. Valid only
            if side bar plots are added. Defaults to 5 for the central CoMut
            and 1 for each side plot.

            Example:
            --------
            widths = [0.5, 5]
            CoMut.plot_comut(widths=heights)

        wspace: float, default 0.2
            The distance between different plots in the x-direction
            (ie side bar plots)

        shadow_width_left: float, default=None
            If not None, width of the shadow subplot positioned left of the central CoMut plot. Useful
            for controlling space for y-axis labels.

        structure: list-like
            List containing desired CoMut structure. Must be provided
            as list of lists (see example). Default structure is to place
            each plot in its own list.

            Example:
            --------
            # plot plot1 and plot2 in a separate subplot from plot4, don't plot
            # plot3.
            structure = [('plot1', 'plot2'), ('plot4')]
            CoMut.plot_comut(structure=structure)

        sublot_hspace: float
            The distance between plots in a subplot. The scale for
            subplot_hspace and hspace are not the same.

        figsize (float, float), optional, default: (10,6)
            width, height of CoMut figure in inches. Only valid if fig argument
            is None.

        Returns:
        -------
        self: CoMut object
            CoMut object with updated axes and figure attributes.

        Example
        --------
        # create CoMut object
        ex_comut = comut.CoMut()

        # add mutation data
        ex_comut.add_categorical_data(mutation_data, name='mutation')

        # add clinical data
        ex_comut.add_categorical_data(tumor_stage, name='tumor_stage')
        ex_comut.add_continuous_data(purity_data, name='purity')

        # plot CoMut data
        ex_comut.plot_comut()

        # ex_comut.axes will be a dictionary with keys 'mutation', 'tumor_stage',
        # and 'purity', with values equal to the plotted axes.'''

        # default structure is each plot to its own subplot
        if structure is None:
            structure = [[plot] for plot in self._plots]

        if heights is None:
            heights = {}

        # define number of subplots
        num_subplots = len(structure)

        # get height structure based on input heights
        heights = self._get_height_spec(structure, heights)

        # calculate height of plots for gridspeccing. Heights are
        # reversed to match CoMut plotting (bottom to top)
        plot_heights = [sum(height) for height in heights][::-1]

        # create figure if none given
        if fig is None:
            fig = plt.figure(figsize=figsize)

        # make default widths and determine location of CoMut. If widths give,
        # just calculate location of CoMut
        if widths is None:
            widths, comut_idx = self._get_default_widths_and_comut_loc()
        else:
            _, comut_idx = self._get_default_widths_and_comut_loc()

        if shadow_width_left is not None:
            shadow_idx_left = comut_idx
            comut_idx += 1
            widths.insert(shadow_idx_left, shadow_width_left)
        else:
            shadow_idx_left = None

        # number of cols is equal to size of widths
        num_cols = len(widths)

        # create gridspec if none given
        if spec is None:
            spec = gridspec.GridSpec(ncols=num_cols, nrows=num_subplots, figure=fig,
                                     height_ratios=plot_heights, width_ratios=widths,
                                     hspace=hspace, wspace=wspace)

        # otherwise, create gridspec in spec
        else:
            spec = gridspec.GridSpecFromSubplotSpec(ncols=num_cols, nrows=num_subplots,
                                                    height_ratios=plot_heights, width_ratios=widths,
                                                    hspace=hspace, wspace=wspace, subplot_spec=spec)

        # plot each plot in subplots
        for i, (plot, height) in enumerate(zip(structure, heights)):
            # subplots share an x axis with first plot
            if i == 0:
                sharex = None
                first_plot = plot[0]
            else:
                sharex = self.axes[first_plot]

            # if only one plot in subplot, just add subplot and plot
            if len(plot) == 1:
                plot_name = plot[0]
                ax = fig.add_subplot(spec[num_subplots - i - 1, comut_idx], sharex=sharex)

                plot_type = self._plots[plot_name]['type']
                data = self._plots[plot_name]['data']
                ax = self._plot_data_on_axis(ax=ax, plot_name=plot_name, plot_type=plot_type,
                                             plot_params=self._plots[plot_name], data=data,
                                             x_padding=x_padding, y_padding=y_padding, tri_padding=tri_padding)

                # extract all overplots on this axis
                over_plots = self._over_plots[plot_name]
                for over_name, over_plot in over_plots.items():
                    plot_type = over_plot['type']
                    data = over_plot['data']
                    ax = self._plot_data_on_axis(ax=ax, plot_name=over_name, plot_type=plot_type,
                                                 plot_params=over_plot, data=data,
                                                 x_padding=x_padding, y_padding=y_padding, tri_padding=tri_padding)

                # extract all sideplots on this axis
                side_plots = self._side_plots[plot_name]

                # identify the locations of each sideplot and plot from inward -> outward
                left_idx, right_idx = 1, 1
                for side_name, side_plot in side_plots.items():
                    position = side_plot['position']
                    if position == 'left':
                        sideplot_idx = comut_idx - left_idx
                        left_idx += 1
                        if sideplot_idx == shadow_idx_left:
                            sideplot_idx -= 1
                            left_idx += 1
                    elif position == 'right':
                        sideplot_idx = comut_idx + right_idx
                        right_idx += 1

                    # sideplots are paired with central CoMut plot
                    side_ax = fig.add_subplot(spec[num_subplots - i - 1, sideplot_idx])
                    side_plot_type = side_plot["type"]
                    side_plot_params = {k:v for k,v in side_plot.items() if k!="type"}

                    if side_plot_type=="bar":
                        side_ax = self._plot_side_bar_data(side_ax, side_name, y_padding=y_padding, **side_plot_params)
                    elif side_plot_type=="error":
                        side_ax = self._plot_side_error_data(side_ax, side_name, y_padding=y_padding, **side_plot_params)
                    else:
                        raise ValueError("Urecognized type {} of side plot."
                                         " Choose either 'bar' or 'error'".format(side_plot_type))

                    # force side axis to match paired axis. Avoiding sharey in case
                    # side bar needs different ticklabels
                    side_ax.set_ylim(ax.get_ylim())

            # if multiple plots in subplot, subplot gridspec required
            else:
                num_plots = len(plot)

                # reverse the heights to be bottom up
                height = height[::-1]
                subplot_spec = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=num_plots,
                                                                hspace=subplot_hspace,
                                                                subplot_spec=spec[num_subplots - i - 1, comut_idx],
                                                                height_ratios=height)
                # plot all data in subplots
                for j, plot_name in enumerate(plot):

                    # plot the data on a subplot within that subgridspec
                    ax = fig.add_subplot(subplot_spec[num_plots - j - 1, 0], sharex=sharex)
                    plot_type = self._plots[plot_name]['type']
                    data = self._plots[plot_name]['data']
                    ax = self._plot_data_on_axis(ax=ax, plot_name=plot_name, plot_type=plot_type,
                                                 plot_params=self._plots[plot_name], data=data,
                                                 x_padding=x_padding, y_padding=y_padding, tri_padding=tri_padding)

                    # side bar plots are not allowed for plots within a subplot
                    if self._side_plots[plot_name]:
                        raise ValueError('Side bar plot for {} cannot be created. '
                                         'Plots within a subplot cannot have a side plot.'.format(plot_name))


        # add x axis labels to the bottom-most axis, make it visible
        xtick_fontdict = self._plots[first_plot]["xtick_fontdict"]
        xtick_style = self._plots[first_plot]["xtick_style"]
        xtick_show = self._plots[first_plot]["xtick_show"]
        xtick_rotation = self._plots[first_plot]["xtick_rotation"]

        if xtick_show:
            self.axes[first_plot].set_xticks(np.arange(0.5, len(self.samples) + 0.5))
            self.axes[first_plot].set_xticklabels(self.samples, fontdict=xtick_fontdict, style=xtick_style,
                                                  rotation=xtick_rotation)
        else:
            self.axes[first_plot].set_xticklabels([])

        self.axes[first_plot].get_xaxis().set_visible(True)
        self.axes[first_plot].tick_params(axis='x', which='both', bottom=False, length=0)

        self.figure = fig
        return self

    def add_axis_legend(self, name, border_white=None, rename=None, order=None,
                        ignored_values=None, title_align='left', bbox_to_anchor=(1, 1),
                        frameon=False, **legend_kwargs):
        '''Add a legend to a named axis of the CoMut plot

        Params:
        -------
        name: str or int
            Name of axis on which to create the legend. Names are created
            when data is added.

        border_white: list-like
            List of categories to replace with a black bordered box.

        rename: dict
            A dictionary for renaming categories. The key should be the
            original category name, with value as the new category name.

        order: list-like
            Order of values in the legend. By default, values are
            sorted alphabetically.

        ignored_values: list-like
            List of values ignored by the legend. Defaults to
            ['Not Available', 'Absent'].

        title_align: str, one of 'left', 'center', or 'right', default 'left'
            The alignment of the legend title in the legend. If no title is
            specified, nothing happens.

        bbox_to_anchor: BboxBase, 2-tuple, or 4-tuple of floats, default (1, 1)
            The location of the legend relative to the axis.

        frameon: bool, default False
            Whether a frame should be drawn around the legend

        legend_kwargs: kwargs
            kwargs to pass to plt.legend()

        Returns:
        --------
        legend: matplotlib legend object
            Legend object created for the input named axis. Can be altered with
            other matplotlib functions that work on legends.'''

        # define defaults
        if border_white is None:
            border_white = []

        if rename is None:
            rename = {}

        if order is None:
            order = []

        if ignored_values is None:
            ignored_values = ['Not Available', 'Absent']

        # define the axis
        axis = self.axes[name]

        plot_type = self._plots[name]['type']
        if plot_type == 'continuous':
            raise ValueError('add_axis_legend is not valid for continuous data.')

        # extract current handles and labels on axis
        if handles is not None and labels is not None:
            handle_lookup = dict(zip(labels, handles))
        else:
            handles, labels = self.axes[name].get_legend_handles_labels()
            handle_lookup = dict(zip(labels, handles))

        # replace borde_white values with a white patch and black border
        for patch_name in border_white:
            handle_lookup[patch_name] = patches.Patch(facecolor='white', edgecolor='black',
                                                      label=border_white)

        # rename categories and delete old ones
        for old_name, new_name in rename.items():
            handle_lookup[new_name] = handle_lookup[old_name]
            del handle_lookup[old_name]

        # ignore certain values
        for value in ignored_values:
            if value in handle_lookup:
                del handle_lookup[value]

        # sort labels by order, reorder handles to match
        sorted_labels = self._sort_list_by_list(handle_lookup.keys(), order)
        sorted_handles = [handle_lookup[label] for label in sorted_labels]

        # create legend
        legend = axis.legend(sorted_handles, sorted_labels, bbox_to_anchor=bbox_to_anchor,
                             frameon=frameon, **legend_kwargs)

        # align legend title
        legend._legend_box.align = title_align
        return legend

    def add_unified_legend(self, axis_name=None, border_white=None, headers=True, headers_separate=[],
                           rename=None, bbox_to_anchor=(1, 1), ignored_values=None,
                           ignored_plots=None, frameon=False, handles_more=None, labels_more=None,
                           titles_more=None, labels_orders={}, ncol=None, nrow=None,  **legend_kwargs):
        '''Add a unified legend to the CoMut plot

        This combines all the various legends into a one column master legend.
        By default, the legend is placed at the top axis.

        Params:
        -------
        axis_name: str or int
            Name of axis on which to create the legend. Names are created
            when data is added.

        border_white: list-like
            List of categories to border for legend entry. This will replace whatever
            patch currently exists with a white box bordered in black.

        headers: bool, default True
            Whether the legend should include subtitles. Subtitles are left
            aligned.

        rename: dict
            A dictionary for renaming categories. The key should be the
            original category name, with value as the new category name.
            Renaming occurs after adding border white patches.

        bbox_to_anchor: BboxBase, 2-tuple, or 4-tuple of floats, default (1,1)
            The location of the legend relative to the axis.

        ignored_values: list-like
            List of ignored values. These categories will be ignored by
            the legend. Defaults to ['Absent', 'Not Available'].

        ignored_plots: list-like
            List of ignored plots. Legends for these plots will not be drawn.

        frameon: bool, default False
            Whether a frame should be drawn around the legend

        handles_more: list-like
            List of lists new handles if any.

        labels_more: list-like
            List of lists new labels if any.

        titles_more: list-like
            List of lists new titles if any.

        labels_orders: dict
            Dict of lists where keys are axes names and values are lists of ordered labels for the axis.

        legend_kwargs: kwargs
            Other kwargs to pass to ax.legend().

        Returns:
        --------
        legend: matplotlib legend object
            Legend object created for the input named axis.'''

        if border_white is None:
            border_white = []

        if rename is None:
            rename = {}

        if ignored_values is None:
            ignored_values = ['Absent', 'Not Available']

        if ignored_plots is None:
            ignored_plots = []

        if handles_more is None:
            handles_more = []

        if labels_more is None:
            labels_more = []

        if titles_more is None:
            titles_more = []

        # store labels and patches
        legend_labels = []
        legend_patches = []
        legend_headers = []

        # loop through plots in reverse order (since plots are bottom up)
        plot_names = list(self._plots.keys())[::-1]
        plot_data_list = list(self._plots.values())[::-1]

        # extract the legend information for each plot and add to storage
        for name, plot_data in zip(plot_names, plot_data_list):
            if name in ignored_plots:
                pass
            else:
                axis = self.axes[name]
                plot_type = plot_data['type']

                if plot_type in ['categorical', 'bar', 'indicator']:
                    # nonstacked bar charts don't need legend labels
                    if plot_type == 'bar' and not plot_data['stacked']:
                        continue

                    handles, labels = axis.get_legend_handles_labels()
                    handles_labels = [(l,h) for l, h in zip(labels, handles)]

                    if name in labels_orders:
                        # if order specified, sort by order given
                        labels_order = labels_orders[name]
                        labels_order_index = {l: i for i, l in enumerate(labels_order)}
                        handles_labels = sorted(handles_labels, key=lambda x: labels_order_index[x[0]])
                    else:
                        # sort alphabetically
                        handles_labels = sorted(handles_labels, key=lambda x: x[0])

                    handles = [x[1] for x in handles_labels]
                    labels = [x[0] for x in handles_labels]

                    # create label-patch dict
                    handle_lookup = dict(zip(labels, handles))

                    # delete ignored categories
                    for value in ignored_values:
                        if value in handle_lookup:
                            del handle_lookup[value]

                    # border the white patches
                    for patch_name in border_white:
                        if patch_name in handle_lookup:
                            handle_lookup[patch_name] = patches.Patch(facecolor='white', edgecolor='black',
                                                                      label=patch_name)

                    # add legend subheader for nonindicator data
                    if plot_type != 'indicator' and headers:
                        legend_labels.append(name)
                        legend_patches.append(patches.Patch(color='none', alpha=0))
                        legend_headers.append(name)

                    # add plot labels and legends
                    legend_labels += list(handle_lookup.keys())
                    legend_patches += list(handle_lookup.values())

                    # add a spacer patch
                    legend_labels.append('')
                    legend_patches.append(patches.Patch(color='white', alpha=0))

                    # rename labels
                    legend_labels = [rename.get(label, label) for label in legend_labels]

        for handles, labels, title in zip(handles_more, labels_more, titles_more):
            # create label-patch dict
            handle_lookup = dict(zip(labels, handles))

            # border the white patches
            for patch_name in border_white:
                if patch_name in handle_lookup:
                    handle_lookup[patch_name] = patches.Patch(facecolor='white', edgecolor='black',
                                                              label=patch_name)

            # add legend header
            if headers:
                legend_labels.append(title)
                legend_patches.append(patches.Patch(color='none', alpha=0))
                legend_headers.append(title)

            # add plot labels and legends
            legend_labels += list(handle_lookup.keys())
            legend_patches += list(handle_lookup.values())

            # add a spacer patch
            legend_labels.append('')
            legend_patches.append(patches.Patch(color='white', alpha=0))

            # rename labels
            legend_labels = [rename.get(label, label) for label in legend_labels]

        # organize the legend if legend labels from some headers should be displayed in separate columns
        if headers_separate is None:
            headers_separate = []
        else:
            headers_separate = [x for x in legend_labels if x in headers_separate]

        if ncol is not None and nrow is not None:
            raise ValueError("You may specify only one of 'ncol' or 'nrow'.")
        elif ncol is not None:
            if len(headers_separate)>0:
                raise NotImplementedError("You may not use headers_separate in combination with ncol. " + \
                                          "Use nrow instead so that the number of columns is automatically set.")
        else:
            if len(headers_separate) > 0 and headers==True:
                if nrow is None:
                    # determine the largest number of rows necessary to draw headers in separate columns
                    nrow = 0
                    for header_separate in headers_separate:
                        # get the start and end indices of legend items attached to the header
                        start, i_start = False, 0
                        stop, i_stop = False, len(legend_labels)-1
                        for i, legend_label in enumerate(legend_labels):
                            if not start:
                                if legend_label==header_separate:
                                    start = True
                                    i_start = i
                            else:
                                if legend_label in legend_headers:
                                    i_stop = i-1
                                    break
                        if i_stop - i_start > nrow:
                            nrow = i_stop - i_start

                if nrow < 2:
                    raise ValueError("The value of nrow cannot be smaller than 2.")

                for header_separate in headers_separate:
                    # get the start and end indices of legend items attached to the header
                    start, i_start = False, 0
                    stop, i_stop = False, len(legend_labels)-1
                    for i, legend_label in enumerate(legend_labels):
                        if not start:
                            if legend_label==header_separate:
                                start = True
                                i_start = i
                        else:
                            if legend_label in legend_headers:
                                i_stop = i-1
                                break

                    while i_start % nrow != 0:
                        # add a spacer patch
                        legend_labels.insert(i_start-1, '')
                        legend_patches.insert(i_start-1,patches.Patch(color='white', alpha=0))
                        i_start += 1
                        i_stop += 1

                    if (i_stop - i_start + 1) % nrow == 0:
                        if i_stop < len(legend_labels)-1 and legend_labels[i_stop+1] == '':
                            del legend_labels[i_stop+1]
                            del legend_patches[i_stop+1]
                    else:
                        if i_stop - i_start + 1 > nrow:
                            if legend_labels[i_stop] == '':
                                del legend_labels[i_stop]
                                del legend_patches[i_stop]
                                i_stop = i_stop - 1

                        while (i_stop - i_start + 1) % nrow != 0:
                            # add a spacer patch
                            legend_labels.insert(i_stop+1, '')
                            legend_patches.insert(i_stop+1,patches.Patch(color='white', alpha=0))
                            i_stop += 1

            nlabs = len(legend_labels)

            if nrow is None:
                nrow = nlabs
                ncol = 1
            elif nrow < 2:
                raise ValueError("The value of nrow cannot be smaller than 2.")
            else:
                ncol = int(np.ceil(nlabs/nrow))

            while nlabs<ncol*nrow:
                # add a spacer patch
                legend_labels.append('')
                legend_patches.append(patches.Patch(color='white', alpha=0))
                nlabs += 1

        # add to the top axis if no axis is given
        if axis_name is None:
            axis = self.axes[list(self.axes.keys())[-1]]
        else:
            axis = self.axes[axis_name]

        # add legend to axis
        leg = axis.legend(labels=legend_labels, handles=legend_patches,
                          bbox_to_anchor=bbox_to_anchor, frameon=frameon, ncol=ncol, **legend_kwargs)

        # more involved code to align the headers
        if headers:
            vpackers = leg.findobj(offsetbox.VPacker)
            for vpack in vpackers[:-1]:  # Last vpack will be the title box
                vpack.align = 'left'
                for hpack in vpack.get_children():
                    draw_area, text_area = hpack.get_children()
                    for collection in draw_area.get_children():
                        alpha = collection.get_alpha()

                        # if the patch has alpha = 0, set it to invisible,
                        # which will shift over it's label to align.
                        if alpha == 0:
                            draw_area.set_visible(False)

        return leg
