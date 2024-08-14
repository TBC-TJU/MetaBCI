import matplotlib as mpl
import mne
import numpy as np
mpl.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MyFigure(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.axes = self.fig.add_subplot(111)

    def mne_EEG_plot(self, data, ch_pos, range):
        self.axes.clear()
        im, cn = mne.viz.plot_topomap(data=data,
                                      pos=np.array(ch_pos) / 1000,
                                      names=None,
                                      axes=self.axes,
                                      sensors=True,
                                      show=False,
                                      cmap='coolwarm',
                                      vlim=[-range, range]
        )
        self.draw()

