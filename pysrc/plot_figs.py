'''
File: plot_figs.py
Project: meta_paper_plot
File Created: Monday, 20th December 2021 9:37:34 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Sunday, 31st July 2022 7:37:27 pm
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2021 - 2022 Amru, University of Pennsylvania

Summary: Fill In
'''

#%%
from os import popen, makedirs, system, walk
from os.path import join, isfile, isdir, basename, dirname, exists
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import os
from os.path import join
from cycler import cycler
import random
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import FixedFormatter
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
from sklearn.metrics import pairwise_distances
from statsmodels.graphics import tsaplots
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

#Options
from cycler import cycler
fig_width_pt = 246 #320 #/2 or 510   # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          #'axes.labelsize': 10,
          'font.size': 9,
          'font.weight': "normal",
          "font.family": "serif",
          "font.serif": ['Computer Modern Roman'],
          'legend.fontsize': 6,
          'xtick.labelsize': 7.5,
          'ytick.labelsize': 7.5,
          'xtick.major.width': 0.4,
          'xtick.minor.width': 0.3,
          'ytick.major.width': 0.4,
          'ytick.minor.width': 0.3,
          'text.usetex': True,
          'axes.linewidth': 0.5,
          'axes.prop_cycle': cycler(color='bgrcmyk'),
          'figure.figsize': fig_size
          }
plt.rcParams.update(params)

colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "blue", "green", "red"]
markers = ["s", "o", "^"]
s = [14.5, 15, 15.5]

data_foldername = "Data/"
data_type_foldername = "prelim_data/"

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

d = 3

data_foldername = "Data/"
data_type_foldername = "SGM_data/" 

#######################################################################################

# Fig SGM cluster

data_type_foldername = "SGM_data/"

system_foldername = "Weibull/"
N_foldername = "N=342/"

type_foldername = "minima_exploration/"

PATH = join(data_foldername, data_type_foldername, system_foldername, N_foldername, type_foldername)

cluster_filename = "clusters_74.9.txt"
df = pd.read_csv(join(PATH, cluster_filename), sep=r"\s+",
                                header=None, skiprows=1)
df.columns = ["bias_U_sigma"] + ["uhat_"+str(i) for i in range(3)] + ["ncluster_"+str(i) for i in range(3)]

fig = plt.figure()
fig.set_figheight(2*fig_height)
fig.set_figwidth(fig_width)

ax1 = plt.subplot2grid(shape=(2, 1), loc=(0, 0), colspan=1)
ax2 = plt.subplot2grid(shape=(2, 1), loc=(1, 0), colspan=1, sharex=ax1)

for i in np.arange(3):
    x1, y1 = df[df.columns[0]], df[df.columns[1+2*i]]
    x2, y2 = df[df.columns[0]], df[df.columns[2+2*i]]

    ax1.scatter(x1, y1, s=s[i], color=colors[6+i], marker = markers[i], facecolors='none')
    ax2.scatter(x2, y2, s=s[i], color=colors[6+i], marker = markers[i], facecolors='none')

ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.set_xlabel(r"$U_{\sigma}$")
ax1.set_ylabel(r"$||<\hat{u}>||$")

ax1.set_ylim(bottom=0.12)
im = plt.imread(join(input_foldername, graphs_foldername, "FIG_cluster_0_1_1.png"))
ax1.add_artist(AnnotationBbox(OffsetImage(im, zoom=0.07), (0.85, 0.191), frameon=False))
ax1.annotate(text='Weak Drift', xy=(0.22, 0.126), xycoords='data', fontsize=7, ha='center')
ax1.annotate(text='Strong Drift', xy=(3.6e0, 0.126), xycoords='data', fontsize=7, ha='center')

ax2.set_xlabel(r"$\mathcal{U}_{\sigma}$")
ax2.set_ylabel(r"$N_{clusters}$")
ax2.set_xscale("log")
ax2.set_yscale("log") 

im = plt.imread(join(input_foldername, graphs_foldername, "FIG_cluster_0_2_1.png"))
ax2.add_artist(AnnotationBbox(OffsetImage(im, zoom=0.07), (0.85, 1.6e2), frameon=False))
ax2.annotate(text='Multiple Clusters', xy=(0.22, 2.09e1), xycoords='data', fontsize=7, ha='center')
ax2.annotate(text='Single Cluster', xy=(3.6e0, 2.09e1), xycoords='data', fontsize=7, ha='center')
fig.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.16)

fig.savefig(output_foldername + graphs_foldername +
            "FIG_SGM_cluster.jpg", dpi=1000, bbox_inches='tight')

# #######################################################################################
