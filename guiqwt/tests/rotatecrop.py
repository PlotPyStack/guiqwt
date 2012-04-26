# -*- coding: utf-8 -*-
#
# Copyright © 2012 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Rotate/crop test: using the scaler C++ engine to resize images"""

SHOW = True # Show test in GUI-based test launcher

import os.path as osp

from guiqwt.builder import make
from guiqwt.plot import ImageDialog
from guiqwt.rotatecrop import RotateCropDialog

def test():
    """Image selection test"""
    from guidata import qapplication
    qapp = qapplication()  # analysis:ignore
    
    item = make.trimage(filename=osp.join(osp.dirname(__file__), "brain.png"))
    dlg = RotateCropDialog(None)
    dlg.set_item(item)
    if dlg.exec_():
        item = dlg.get_item()
        dlg2 = ImageDialog()
        plot = dlg2.get_plot()
        plot.add_item(item)
        dlg2.exec_()
    
if __name__ == "__main__":
    test()
