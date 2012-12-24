# -*- coding: utf-8 -*-
#
# Copyright © 2010 CEA
# Ludovic Aubry
# Licensed under the terms of the CECILL License
# (see guidata/__init__.py for details)

"""CurvePlotDialog test"""


#===============================================================================
#TODO: Make this test work!!
#===============================================================================



SHOW = False # Show test in GUI-based test launcher

from guidata.qt.QtGui import QFont

from guiqwt.baseplot import BasePlot
from guiqwt.plot import CurveDialog, CurveWidget, PlotManager
from guiqwt.builder import make
from guiqwt.curve import CurvePlot


class MyPlotDialog(CurveDialog):
    def create_plot(self, options):
        manager = PlotManager(None)
        self.plotwidget = CurveWidget(self, manager=manager, **options)
        manager.set_main(self.plotwidget)
        plot1 = CurvePlot(title="TL")
        plot2 = CurvePlot(title="TR")
        plot3 = CurvePlot(title="BL")
        plot4 = CurvePlot(title="BR")
        self.plotwidget.add_plot(plot1, 0, 0, "1")
        self.plotwidget.add_plot(plot2, 0, 1, "2")
        self.plotwidget.add_plot(plot3, 1, 0, "3")
        self.plotwidget.add_plot(plot4, 1, 1, "4")
        self.plotwidget.finalize()
        manager.synchronize_axis(BasePlot.X_BOTTOM, ["1", "3"])
        manager.synchronize_axis(BasePlot.X_BOTTOM, ["2", "4"])
        manager.synchronize_axis(BasePlot.Y_LEFT,   ["1", "2"])
        manager.synchronize_axis(BasePlot.Y_LEFT,   ["3", "4"])
        
        self.layout.addWidget(self.plotwidget, 0, 0)

def plot(items1, items2, items3, items4):
    win = MyPlotDialog(edit=False, toolbar=True,
                       wintitle="CurvePlotDialog test",
                       options=dict(title="Title", xlabel="xlabel",
                                    ylabel="ylabel"))
    items = [items1, items2, items3, items4]
    for i, plot in enumerate(win.plotwidget.plots):
        for item in items[i]:
            plot.add_item(item)
        plot.set_axis_font("left", QFont("Courier"))
        plot.set_items_readonly(False)
    win.get_panel("itemlist").show()
    win.show()
    win.exec_()

def test():
    """Test"""
    # -- Create QApplication
    import guidata
    _app = guidata.qapplication()
    # --
    from numpy import linspace, sin
    x = linspace(-10, 10, 200)
    dy = x/100.
    y = sin(sin(sin(x)))    
    x2 = linspace(-10, 10, 20)
    y2 = sin(sin(sin(x2)))
    plot([make.curve(x, y, color="b"),
          make.label("Relative position <b>outside</b>",
                     (x[0], y[0]), (-10, -10), "BR"),],
         [make.curve(x2, y2, color="g"),
          ],
         [make.curve(x, sin(2*y), color="r"),
          make.label("Relative position <i>inside</i>",
                     (x[0], y[0]), (10, 10), "TL"),],
         [make.merror(x, y/2, dy),
          make.label("Absolute position", "R", (0, 0), "R"),
          make.legend("TR"),]
         )

if __name__ == "__main__":
    test()
