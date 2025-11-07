import seaborn as sns
import numpy as np
import utils
import config as c
from plots.video import record_video

sns.set_theme(style='darkgrid', font_scale=3.0)


def main():
    width = 6

    # Parse arguments
    options = utils.get_plot_options()

    # Load log
    log = np.load('simulation/log_{}.npz'.format(c.log_name))

    # Choose plot to display
    record_video(log, width)


if __name__ == '__main__':
    main()
