# -*- coding: utf-8 -*-
#
# Copyright © 2010 Ludovic Aubry
# Copyright © 2022 Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guidata/__init__.py for details)

"""SyncPlotDialog test"""

import numpy as np
from guidata.configtools import get_icon
from guidata.qthelpers import win32_fix_title_bar_background
from guiqwt.baseplot import BasePlot
from guiqwt.builder import make
from guiqwt.config import _
from guiqwt.curve import CurvePlot, CurveItem
from guiqwt.image import ImagePlot
from guiqwt.plot import PlotManager, SubplotWidget
from guiqwt.tests.image_coords import create_2d_gaussian
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy import QtCore as QC

# guitest: show


class SyncPlotWindow(QW.QMainWindow):
    """Window for showing plots, optionally synchronized"""

    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.setWindowTitle(self.__doc__ if title is None else title)
        self.setWindowIcon(get_icon("guiqwt.svg"))
        self.manager = PlotManager(None)
        self.manager.set_main(self)
        self.subplotwidget = SubplotWidget(self.manager, parent=self)
        self.setCentralWidget(self.subplotwidget)
        toolbar = QW.QToolBar(_("Tools"))
        self.manager.add_toolbar(toolbar, "default")
        toolbar.setMovable(True)
        toolbar.setFloatable(True)
        self.addToolBar(toolbar)

    def add_panels(self):
        """Add standard panels"""
        self.subplotwidget.add_standard_panels()

    def rescale_plots(self):
        """Rescale all plots"""
        QW.QApplication.instance().processEvents()
        for plot in self.subplotwidget.plots:
            plot.do_autoscale()

    def showEvent(self, event):  # pylint: disable=C0103
        """Reimplement Qt method"""
        super().showEvent(event)
        QC.QTimer.singleShot(0, self.rescale_plots)

    def add_plot(self, row, col, plot, sync=False, plot_id=None):
        """Add plot to window"""
        if plot_id is None:
            plot_id = str(len(self.subplotwidget.plots) + 1)
        self.subplotwidget.add_subplot(plot, row, col, plot_id)
        if sync and len(self.subplotwidget.plots) > 1:
            syncaxis = self.manager.synchronize_axis
            for i_plot in range(len(self.subplotwidget.plots) - 1):
                syncaxis(BasePlot.X_BOTTOM, [plot_id, f"{i_plot + 1}"])
                syncaxis(BasePlot.Y_LEFT, [plot_id, f"{i_plot + 1}"])


def plot(*itemlists):
    """Plot items in SyncPlotDialog"""
    win = SyncPlotWindow()
    row, col = 0, 0
    has_curves = any(isinstance(item, CurveItem) for item in itemlists[0])
    for items in itemlists:
        plot = CurvePlot() if has_curves else ImagePlot()
        for item in items:
            plot.add_item(item)
        plot.set_axis_font("left", QG.QFont("Courier"))
        plot.set_items_readonly(False)
        win.add_plot(row, col, plot, sync=True)
        col += 1
        if col == 2:
            row += 1
            col = 0
    win.add_panels()
    contrast = win.subplotwidget.contrast
    if contrast is not None:
        contrast.show()
    win.resize(800, 600)
    win.show()


def test_curves():
    """Test"""
    x = np.linspace(-10, 10, 200)
    dy = x / 100.0
    y = np.sin(np.sin(np.sin(x)))
    x2 = np.linspace(-10, 10, 20)
    y2 = np.sin(np.sin(np.sin(x2)))
    plot(
        [
            make.curve(x, y, color="b"),
            make.label(
                "Relative position <b>outside</b>", (x[0], y[0]), (-10, -10), "BR"
            ),
        ],
        [
            make.curve(x2, y2, color="g"),
        ],
        [
            make.curve(x, np.sin(2 * y), color="r"),
            make.label("Relative position <i>inside</i>", (x[0], y[0]), (10, 10), "TL"),
        ],
        [
            make.merror(x, y / 2, dy),
            make.label("Absolute position", "R", (0, 0), "R"),
            make.legend("TR"),
        ],
    )
    QW.QApplication.instance().exec()


def test_images():
    """Test"""
    img1 = create_2d_gaussian(20, np.uint8, x0=-10, y0=-10, mu=7, sigma=10.0)
    img2 = create_2d_gaussian(20, np.uint8, x0=-10, y0=-10, mu=5, sigma=8.0)
    img3 = create_2d_gaussian(20, np.uint8, x0=-10, y0=-10, mu=3, sigma=6.0)

    def makeim(data):
        """Make image item"""
        return make.image(data, interpolation="nearest")

    plot([makeim(img1)], [makeim(img2)], [makeim(img3)])
    QW.QApplication.instance().exec()


if __name__ == "__main__":
    # -- Create QApplication
    import guidata

    _APP = guidata.qapplication()
    test_curves()
    test_images()
