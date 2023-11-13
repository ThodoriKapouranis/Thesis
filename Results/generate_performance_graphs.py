'''
This file generates the graphs used for comparing the Water IoU performance between 
same architectures but different speckle filters.

Generate for : XGB, UNet
Generate for : in-region, out-of-region
---
Y-axis : Water IoU (%)
X-axis : Scenario 1,2,3 ticks
Labels : base, mean, lee, 
'''

from absl import app, flags
import os
from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.append("../Thesis")

FILTER_NAMES = ["None", "Mean", "Lee", "RLF", "Frost"]

FILTER_COLORS = {
    FILTER_NAMES[0]: "grey",
    FILTER_NAMES[1]: "green",
    FILTER_NAMES[2]: "orange",
    FILTER_NAMES[3]: "red",
    FILTER_NAMES[4]: "blue"
}

FILTER_MARKERS = {
    FILTER_NAMES[0]: ".",
    FILTER_NAMES[1]: "s",
    FILTER_NAMES[2]: "x",
    FILTER_NAMES[3]: "P",
    FILTER_NAMES[4]: "*"
    }

UNET_IOU = {
    'in': {
        1: {
            FILTER_NAMES[0]: 58.066,
            FILTER_NAMES[1]: 58.544,
            FILTER_NAMES[2]: 58.564,
            FILTER_NAMES[3]: 59.523,
            FILTER_NAMES[4]: 59.726,
        },
        2: {
            FILTER_NAMES[0]: 58.159,
            FILTER_NAMES[1]: 57.281,
            FILTER_NAMES[2]: 58.784,
            FILTER_NAMES[3]: 58.69,
            FILTER_NAMES[4]: 58.254,
        },
        3: {
            FILTER_NAMES[0]: 59.263,
            FILTER_NAMES[1]: 60.351,
            FILTER_NAMES[2]: 60.237,
            FILTER_NAMES[3]: 60.271,
            FILTER_NAMES[4]: 59.344,
        },
    },
    'out':{
        1: {
            FILTER_NAMES[0]: 58.491,
            FILTER_NAMES[1]: 58.937,
            FILTER_NAMES[2]: 59.189,
            FILTER_NAMES[3]: 58.777,
            FILTER_NAMES[4]: 58.033,
        },
        2: {
            FILTER_NAMES[0]: 60.062,
            FILTER_NAMES[1]: 58.386,
            FILTER_NAMES[2]: 60.503,
            FILTER_NAMES[3]: 60.312,
            FILTER_NAMES[4]: 59.889,
        },
        3: {
            FILTER_NAMES[0]: 60.689,
            FILTER_NAMES[1]: 60.948,
            FILTER_NAMES[2]: 60.959,
            FILTER_NAMES[3]: 61.043,
            FILTER_NAMES[4]: 55.692,
        }
}
}

XGB_IOU = {
    'in':{
        1: {
            FILTER_NAMES[0]: 46.383,
            FILTER_NAMES[1]: 48.903,
            FILTER_NAMES[2]: 42.924,
            FILTER_NAMES[3]: 50.071,
            FILTER_NAMES[4]: 48.955,
        },
        2: {
            FILTER_NAMES[0]: 48.337,
            FILTER_NAMES[1]: 47.929,
            FILTER_NAMES[2]: 48.502,
            FILTER_NAMES[3]: 51.409,
            FILTER_NAMES[4]: 49.721,
        },
        3: {
            FILTER_NAMES[0]: 53.359,
            FILTER_NAMES[1]: 53.217,
            FILTER_NAMES[2]: 50.861,
            FILTER_NAMES[3]: 53.733,
            FILTER_NAMES[4]: 54.814,
        },
    },
    'out':{
        1: {
            FILTER_NAMES[0]: 36.736,
            FILTER_NAMES[1]: 45.133,
            FILTER_NAMES[2]: 41.513,
            FILTER_NAMES[3]: 47.909,
            FILTER_NAMES[4]: 45.163,
        },
        2: {
            FILTER_NAMES[0]: 41.919,
            FILTER_NAMES[1]: 42.761,
            FILTER_NAMES[2]: 49.467,
            FILTER_NAMES[3]: 50.82,
            FILTER_NAMES[4]: 47.424,
        },
        3: {
            FILTER_NAMES[0]: 51.426,
            FILTER_NAMES[1]: 51.542,
            FILTER_NAMES[2]: 51.603,
            FILTER_NAMES[3]: 52.402,
            FILTER_NAMES[4]: 54.35,
        },
    }
}

try:
    os.mkdir("workspaces/Thesis/Results/graphs")
except:
    print("WARNING: Directory already exists")
    pass


arch_names = ["UNet", "XGB"]
for i, results in enumerate([UNET_IOU, XGB_IOU]):
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    for ax, dataset in enumerate(['in', 'out']):
        axes[ax].set_xticks([1,2,3], ["1", "2", "3"])
        axes[ax].set_xlabel("Scenarios")
        axes[ax].set_ylabel("Water IoU (%)")
        axes[ax].set_ylim([35,65])
        axes[ax].set_title(f"{arch_names[i]} water IoU, {dataset}-region dataset")
        
        for filter in FILTER_NAMES:
            x = [1,2,3]
            y = [results[dataset][scenario][filter] for scenario in [1,2,3]]
            
            label = None if dataset=="out" else filter
            axes[ax].scatter(x, y, c=FILTER_COLORS[filter], marker=FILTER_MARKERS[filter], label=label)

            b, a = np.polyfit(x, y, deg=1)
            xrange = np.linspace(1,3, num=100)
            axes[ax].plot(xrange, a + b*xrange, color=FILTER_COLORS[filter], alpha=0.3)
                
    fig.legend()



    fig.savefig(f"/workspaces/Thesis/Results/graphs/{arch_names[i]}")
