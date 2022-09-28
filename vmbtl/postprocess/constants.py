# -*- coding: utf-8 -*-

from matplotlib.colors import TABLEAU_COLORS

FIGURES_DIR = "figures"

IGNORE_NNS = [2, 9]
SORTED_NNS = sorted([str(i) for i in range(16)])
IGNORE_IDX = [SORTED_NNS.index(str(i)) for i in IGNORE_NNS]

TABLEAU_COLORS = [color[1] for color in TABLEAU_COLORS.items()]
