import os
import matplotlib
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

DARK_MODE = False
BACKGROUND_COLOR = "#1E1332" if DARK_MODE else "white"
FONT_COLOR = "white" if DARK_MODE else "black"

# TODO: Are these needed as variables?!
PURPLE_COLOR = "#762a83"
ORANGE_COLOR = "#ff4d00"
PLOT_COLOR = PURPLE_COLOR if not DARK_MODE else ORANGE_COLOR


def config_plotting_environment(change_plot_frame=True):
    # Set the font of the general labels to 7.
    # Note: This generally works for my *CL papers,
    # for which the default caption font size is 10.
    font = {"size": 7}
    matplotlib.rc("font", **font)
    matplotlib.rc("text", usetex=False)

    if change_plot_frame:
        # Remove some of the sides of the plot's frame
        matplotlib.rcParams["axes.spines.left"] = True
        matplotlib.rcParams["axes.spines.right"] = False
        matplotlib.rcParams["axes.spines.top"] = False
        matplotlib.rcParams["axes.spines.bottom"] = True

    # Allow using tex in labels and titles
    # Note: This requires the system to have some tex dependencies installed
    matplotlib.rcParams["text.latex.preamble"] = [r"\boldmath"]

    plt.rcParams["axes.facecolor"] = BACKGROUND_COLOR
    # Presentation color:
    plt.rcParams["figure.facecolor"] = BACKGROUND_COLOR
