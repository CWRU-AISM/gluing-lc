# Shared figure style for the paper figures (Liberation Sans, Okabe-Ito palette, 300 DPI).

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SINGLE_COL_WIDTH = 3.25
FULL_WIDTH = 6.5

BLUE = '#0072B2'
ORANGE = '#E69F00'
GREEN = '#009E73'
SKY = '#56B4E9'
PURPLE = '#CC79A7'
RED = '#D55E00'
YELLOW = '#F0E442'
GRAY = '#999999'
LGRAY = '#CCCCCC'

MODEL_COLORS = {
    'GPT-2': BLUE,
    'Mistral-7B': ORANGE,
    'Llama-3-8B': GREEN,
    'Llama-2-7B': SKY,
    'Llama-2-13B': PURPLE,
}

METHOD_COLORS = {
    'random': GRAY,
    'h1_dims': BLUE,
    'variance': ORANGE,
    'full_style': PURPLE,
    'h0_removed': GREEN,
    'fisher': RED,
    'Random': GRAY,
    r"$H^1$ (sheaf)": BLUE,
    r"Variance top-$k$": ORANGE,
    'Full style / CAA': PURPLE,
    r"$H^0$-removed": GREEN,
    'Fisher': RED,
}

METHOD_MARKERS = {
    'random': 'o',
    'h1_dims': 's',
    'variance': '^',
    'full_style': 'D',
    'h0_removed': 'v',
    'fisher': 'X',
}

METHOD_LABELS = {
    'random': 'Random',
    'h1_dims': r"$H^1$ (sheaf)",
    'variance': r"Variance top-$k$",
    'full_style': 'Full style / CAA',
    'h0_removed': r"$H^0$-removed",
    'fisher': 'Fisher',
}

BLUE_LIGHT = (0.0, 0.45, 0.70, 0.12)
ORANGE_LIGHT = (0.90, 0.62, 0.0, 0.12)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / 'outputs'


def apply_style():
    # Configure rcParams for ICLR / ICML camera-ready figures.
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'text.usetex': False,
    })


def save_figure(fig, name: str, out_dir: Path = DEFAULT_OUT_DIR, pad_inches: float = 0.05):
    # Save figure as both PNG and PDF in `out_dir`.
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / name
    for ext in ('.png', '.pdf'):
        fig.savefig(
            str(base) + ext,
            facecolor='white',
            bbox_inches='tight',
            pad_inches=pad_inches,
        )
