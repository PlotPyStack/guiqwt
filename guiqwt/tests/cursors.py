# -*- coding: utf-8 -*-
#
# Copyright © 2009-2011 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Horizontal/vertical cursors test"""

SHOW = True # Show test in GUI-based test launcher

from guiqwt.plot import CurveDialog
from guiqwt.builder import make

def plot( *items ):
    win = CurveDialog(edit=False, toolbar=True)
    plot = win.get_plot()
    for item in items:
        plot.add_item(item)
    win.show()
    win.exec_()

def test():
    """Test"""
    # -- Create QApplication
    import guidata
    _app = guidata.qapplication()
    # --
    from numpy import linspace, sin
    x = linspace(-10, 10, 1000)
    y = sin(sin(sin(x)))

    curve = make.curve(x, y, "ab", "b")
    hcursor = make.hcursor(.2)
    vcursor = make.vcursor(2)
    hcursor_info = make.info_cursor(hcursor, "TL")
    vcursor_info = make.info_cursor(vcursor, "BR")
    legend = make.legend("TR")
    plot(curve, hcursor, vcursor, hcursor_info, vcursor_info, legend)

if __name__ == "__main__":
    test()
