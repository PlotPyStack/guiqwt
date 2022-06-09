# -*- coding: utf-8 -*-
#
# Copyright © 2010 Ludovic Aubry
# Copyright © 2022 Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guidata/__init__.py for details)

"""SyncPlotDialog test"""

from guiqwt.config import _
from guidata.configtools import get_icon

from qtpy import QtWidgets as QW
from qtpy import QtGui as QG

from guiqwt.baseplot import BasePlot
from guiqwt.plot import SubplotWidget, PlotManager
from guiqwt.builder import make
from guiqwt.curve import CurvePlot

SHOW = False  # Show test in GUI-based test launcher


class SyncPlotDialog(QW.QDialog):
    """Dialog demonstrating plot synchronization feature"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.__doc__)
        self.setWindowIcon(get_icon("guiqwt.svg"))

        self.manager = manager = PlotManager(None)
        manager.set_main(self)
        self.subplotwidget = spwidget = SubplotWidget(manager, parent=self)
        self.setLayout(QW.QVBoxLayout())
        toolbar = QW.QToolBar(_("Tools"))
        manager.add_toolbar(toolbar)
        self.layout().addWidget(toolbar)
        self.layout().addWidget(spwidget)

        plot1 = CurvePlot(title="TL")
        plot2 = CurvePlot(title="TR")
        plot3 = CurvePlot(title="BL")
        plot4 = CurvePlot(title="BR")
        spwidget.add_subplot(plot1, 0, 0, "1")
        spwidget.add_subplot(plot2, 0, 1, "2")
        spwidget.add_subplot(plot3, 1, 0, "3")
        spwidget.add_subplot(plot4, 1, 1, "4")
        spwidget.add_itemlist()
        manager.synchronize_axis(BasePlot.X_BOTTOM, ["1", "3"])
        manager.synchronize_axis(BasePlot.X_BOTTOM, ["2", "4"])
        manager.synchronize_axis(BasePlot.Y_LEFT, ["1", "2"])
        manager.synchronize_axis(BasePlot.Y_LEFT, ["3", "4"])

        self.manager.register_all_curve_tools()


def plot(items1, items2, items3, items4):
    """Plot items in SyncPlotDialog"""
    dlg = SyncPlotDialog()
    items = [items1, items2, items3, items4]
    for i, plot in enumerate(dlg.subplotwidget.plots):
        for item in items[i]:
            plot.add_item(item)
        plot.set_axis_font("left", QG.QFont("Courier"))
        plot.set_items_readonly(False)
    dlg.manager.get_panel("itemlist").show()
    dlg.show()
    dlg.exec_()


def test():
    """Test"""
    # -- Create QApplication
    import guidata

    _app = guidata.qapplication()
    # --
    from numpy import linspace, sin

    x = linspace(-10, 10, 200)
    dy = x / 100.0
    y = sin(sin(sin(x)))
    x2 = linspace(-10, 10, 20)
    y2 = sin(sin(sin(x2)))
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
            make.curve(x, sin(2 * y), color="r"),
            make.label("Relative position <i>inside</i>", (x[0], y[0]), (10, 10), "TL"),
        ],
        [
            make.merror(x, y / 2, dy),
            make.label("Absolute position", "R", (0, 0), "R"),
            make.legend("TR"),
        ],
    )


if __name__ == "__main__":
    test()
