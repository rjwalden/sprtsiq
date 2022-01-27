from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings


warnings.filterwarnings("ignore")

def corr(df, *cols, size=None):
    '''
    Display correlation matrix for the columns
    '''
    assert len(cols) > 1, 'No columns to display'
    corr = df[list(cols)].corr()
    plt.figure(figsize=size if size is not None else (len(cols),)*2);
    sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap="YlGnBu", alpha=0.7);
    plt.show();


def change_width(ax, new_value) :
    '''
    Changes the width of the bar
    '''
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
    return ax


def perc(ax, total:int, height=None, **kwargs):
    '''
    Annotates barplots with percents for each bar
    '''

    # get each bar
    for p in ax.patches:

        # create label
        label = "{:.1f}%".format(
            100 * p.get_height() / total
        )

        # set label location
        x = p.get_x() + p.get_width() / 2
        y = height if height else p.get_height() / 3

        if y > 0:
            # add label to bar
            ax.annotate(
                label,
                (x, y),
                ha="center",
                va="center",
                size=12,
                xytext=(0, 5),
                textcoords="offset points",
                **kwargs
            )

    # return plot with new labels
    return ax


def univariate_numeric(df:pd.DataFrame, c:str):
    '''
    Performs numeric univariate analysis of a dataframe
    '''
    df[c] = df[c].astype(float) # enforce common type

    # determine skew 
    mean = df[c].mean()
    median = df[c].median()
    if mean - median > 0:
        skew = 'right skewed'
    else:
        skew = 'left skewed'

    # get quartiles
    q1, q3 = df[c].quantile([0.25,0.75])

    # create plot
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), dpi=80, sharex=True)

    # label plot with skew and quartile information
    fig.suptitle(
        f'Numeric: {c}\n[{skew}] [mean: {mean}] [median: {median}] [Q1: {q1}] [ Q3: {q3}]', 
        fontsize=14,
        y=1.025
    )
    ax0 = sns.boxplot(data=df, x=c, ax=axes[0]);
    ax0.xaxis.labelpad = 10
    ax1 = sns.histplot(data=df, x=c, kde=True, ax=axes[1]);
    ax1.xaxis.labelpad = 10
    plt.show();


def univariate_categorical(df:pd.DataFrame, c:str, is_bool=False, rotate=False):
    '''
    Performs categorical univariate analysis of dataframe
    '''

    # enforce common type
    df[c] = df[c].astype('category')

    # get most common record
    top = df[c].describe().loc['top']

    if is_bool:
        top = str(bool(int(top)))


    # add historgram
    ax = sns.histplot(
        df,
        x=c,
        palette='Paired',
        shrink=0.85,
    )

    # add percentiles to the histograms
    ax = perc(ax, len(df))

    # add titles to hitogram plot
    ax.set_title(
        f'Boolean: {c}\n[top: {top}]' if is_bool else f'Category: {c}\n[top: {top}]',
        fontsize=14, 
        pad=24,
    )

    # configure boolean specific xticks
    if is_bool:
        ax.set_xticks(range(2))
        ax.set_xticklabels(['False','True'])

    ax.xaxis.labelpad = 10
    ax.tick_params(axis='x', which='major', pad=5)

    if rotate:
        plt.xticks(rotation=30)

    # space the subplots appropriately
    plt.subplots_adjust(hspace = 0.4)
    plt.show();


def univariate_categorical_circular(
    df:pd.DataFrame, 
    c:str, circle_ratio=0.5,
    sort='count', cmap:str='cool',
    labelPadding = 10,
):
    '''
    Creates a circular bar chart 
    '''
    # Build a dataset
    df = df[c].value_counts().to_frame()
    df['Name'] = df.index.values

    if np.array_equal(df['Name'].dropna(), df['Name'].dropna().astype(int)):
        df['Name'] = df['Name'].astype(int)

    # Reorder the dataframe
    if sort == 'count':
        df = df.sort_values(by=[c])
    elif sort == 'value':
        df = df.sort_values(by=['Name'])


    # initialize the figure
    plt.figure(figsize=(20,10))
    ax = plt.subplot(111, polar=True)
      # add titles to hitogram plot
    ax.set_title(
        f'Catgorical: {c}',
        fontsize=14, 
        pad=24,
    )
    plt.axis('off')

    # Compute max and min in the dataset
    max = df[c].max()
    min = max*circle_ratio

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    slope = (max - min) / max
    heights = slope * df[c] + min

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(df.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index)+1))
    angles = [element * width for element in indexes]

    # Generate hue for bars
    try:
        cmap = getattr(plt.cm, cmap)
    except:
        print('invalid cmap, using default')
        cmap = getattr(plt.cm, 'cool')

    colors=cmap((df[c]-min)/(max-min))

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=min,
        linewidth=2, 
        alpha=0.7,
        edgecolor="black",
        color=colors # "#61a4b2",
    )

    names = df["Name"]

    # Add labels
    for bar, angle, height, label in zip(bars, angles, heights, names):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=min + bar.get_height() + labelPadding, 
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 
    plt.show();


def numeric_binary_bivariate(df:pd.DataFrame, c:str, target:str):
    '''
    Chart bivariate analysis of a column against a binary target with a regplot and a point plot
    '''
    assert c != target

    # enforce types
    df[c] = df[c].astype(float)
    df[target] = df[target].astype(int)

    # point biserial correlation specifically for continuous vairiables correlation with binary column
    res = stats.pointbiserialr(df[target], df[c])

    # create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=80)
    fig.suptitle(
        f'Numeric: {c} vs {target}\n[Point Biserial Correlation: {res.correlation}]', 
        fontsize=14, 
        y=1.025
    )

    # add logistic regplot and point plot to the subplots
    sns.regplot(data=df, x=c, y=target, y_jitter=.1, logistic=True, ax=axes[0]);
    sns.pointplot(data=df, x=target, y=c, ax=axes[1])
    plt.show();


def categorical_binary_bivariate(df:pd.DataFrame, c:str, target:str, is_bool=False, rotate=False):
    '''
    Chart bivariate of a column against a binary target with count plot
    '''
    assert c != target

    plt.locator_params(axis='x', nbins=len(df[c].unique()))

    # make a count plot with respect to the target and current column
    ax = sns.histplot(
        df,
        x=c,
        hue=target,
        palette='Paired',
        shrink=0.85
    )

    # add percent labels to bars
    ax = perc(ax, len(df))

    # add title to subplot 
    ax.set_title(
        f'Category: {c} vs {target}' if not is_bool else  f'Boolean: {c} vs {target}',
        fontsize=14, 
        y=1.05
    )

    # boolean specific xticks
    if is_bool:
        ax.set_xticks(range(2))
        ax.set_xticklabels(['False','True'])

    if rotate:
        plt.xticks(rotation=30)

    plt.show();
