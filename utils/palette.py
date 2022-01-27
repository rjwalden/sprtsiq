import seaborn as sns

background = '#E6E6E6'
colors = ['#1BC3FB', '#003594', '#1C397F']
critical_colors = ['#15DB95', '#AA0000']
binary_colors = ['#003594',  '#AA0000']

sns.set(rc={
  'axes.facecolor': background, 
  'grid.color': '.6', 
  'axes.edgecolor': '.6', 
  'legend.facecolor': 'white', 
  'legend.frameon': True, 
  'text.color': 'black', 
  'axes.labelcolor': 'black', 
  'axes.titlecolor': 'black',
  'xtick.color': 'black',
  'ytick.color': 'black',
  'axes.labelpad': 16,
  'axes.titlepad': 32
})

sns.set_palette(sns.color_palette(colors, desat=1.), 2)