# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Load/save test"""

SHOW = True # Show test in GUI-based test launcher

# WARNING:
# This script requires read/write permissions on current directory

from guidata.qt.QtGui import QFont

import os, os.path as osp
from numpy import linspace, sin

from guiqwt.plot import ImageDialog
from guiqwt.builder import make
from guiqwt.shapes import Axes
from guiqwt.tools import LoadItemsTool, SaveItemsTool

FNAME = "loadsavecanvas.gui"

def build_items():
    x = linspace(-10, 10, 200)
    y = sin(sin(sin(x)))
    filename = osp.join(osp.dirname(__file__), "brain.png")
    items = [ make.curve(x, y, color="b"),
              make.image(filename=filename),
              make.trimage(filename=filename),
              make.label("Relative position <b>outside</b>",
                         (x[0], y[0]), (-10, -10), "BR"),
              make.label("Relative position <i>inside</i>",
                         (x[0], y[0]), (10, 10), "TL"),
              make.label("Absolute position", "R", (0,0), "R"),
              make.legend("TR"),
              make.rectangle(-3, -0.8, -0.5, -1., "rc1"),
              make.ellipse(-10, 0.0, 0, 0, .5, "el1"),
              make.annotated_rectangle(0.5, 0.8, 3, 1., "rc1", "tutu"),
              make.annotated_segment(-1,-1, 1, 1., "rc1", "tutu"),
              Axes( (0,0), (1,0), (0,1) ),
              ]
    return items

def test():
    import guidata
    _app = guidata.qapplication()
    win = ImageDialog(edit=False, toolbar=True, wintitle="Load/save test",
                      options=dict(title="Title", xlabel="xlabel",
                                   ylabel="ylabel"))
    win.add_separator_tool()
    win.add_tool(LoadItemsTool)
    win.add_tool(SaveItemsTool)
    plot = win.get_plot()

    if os.access(FNAME, os.R_OK):
        f = file(FNAME, "rb")
        print "Restoring items...",
        plot.restore_items(f)
        print "OK"
    else:
        for item in build_items():
            plot.add_item(item)
        print "Building items and add them to plotting canvas",
    plot.set_axis_font("left", QFont("Courier"))
    win.get_itemlist_panel().show()
    plot.set_items_readonly(False)
    win.show()
    win.exec_()
    f = file(FNAME, "wb")
    print "Saving items...",
    plot.save_items(f)
    print "OK"

if __name__ == "__main__":
    test()
