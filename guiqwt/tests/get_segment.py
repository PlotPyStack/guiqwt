# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
AnnotatedSegmentTool test

This guiqwt tool provide a MATLAB-like "ginput" feature.
"""

SHOW = True # Show test in GUI-based test launcher

import os.path as osp

from guiqwt.plot import ImageDialog
from guiqwt.tools import SelectTool, AnnotatedSegmentTool
from guiqwt.builder import make
from guiqwt.config import _


def test_function(shape):
    print "Current coordinates:", shape.get_rect()

def get_segment(item):
    """Show image and return selected segment coordinates"""
    win = ImageDialog(_("Select a segment then press OK to accept"), edit=True)
    default = win.add_tool(SelectTool)
    win.set_default_tool(default)
    segtool = win.add_tool(AnnotatedSegmentTool, title="Test",
                           setup_shape_cb=test_function,
                           switch_to_default_tool=True)
    segtool.activate()
    plot = win.get_plot()
    plot.add_item(item)
    plot.set_active_item(item)
    win.show()
    if win.exec_():
        shape = segtool.get_last_final_shape()
        return shape.get_rect()

def test():
    """Test"""
    # -- Create QApplication
    import guidata
    _app = guidata.qapplication()
    # --
    filename = osp.join(osp.dirname(__file__), "brain.png")
    image = make.image(filename=filename, colormap="bone")
    print get_segment(image)

if __name__ == "__main__":
    test()
