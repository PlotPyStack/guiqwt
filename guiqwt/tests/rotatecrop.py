# -*- coding: utf-8 -*-
#
# Copyright © 2012 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Rotate/crop test: using the scaler C++ engine to rotate/crop images"""

SHOW = True # Show test in GUI-based test launcher

import os.path as osp

from guiqwt.builder import make
from guiqwt.plot import ImageDialog
from guiqwt.rotatecrop import RotateCropDialog
from guiqwt.io import imagefile_to_array


def imshow(data, title=None, hold=False):
    dlg = ImageDialog(wintitle=title)
    dlg.get_plot().add_item(make.image(data))
    if hold:
        dlg.show()
    else:
        dlg.exec_()

def test(interactive=True, fname="brain.png"):
    array0 = imagefile_to_array(osp.join(osp.dirname(__file__), fname),
                                to_grayscale=True)
    
    item = make.trimage(array0, dx=.1, dy=.1)
    dlg = RotateCropDialog(None)
    dlg.set_item(item)
    if interactive:
        ok = dlg.exec_()
    else:
        dlg.show()
        dlg.accept()
        ok = True
    if ok:
        array1 = dlg.cropped_array
        if array0.shape == array1.shape:
            if (array1 == array0).all() and not interactive:
                print "Test passed successfully."
                return
            imshow(array1-array0, title="array1-array0")
        else:
            print array0.shape, '-->', array1.shape
        imshow(array0, title="array0", hold=True)
        imshow(array1, title="array1")


if __name__ == '__main__':
    from guidata import qapplication
    qapp = qapplication()  # analysis:ignore
    
    test(interactive=False, fname="brain.png")
#    test(interactive=False, fname="contrast.png")
    test(interactive=True)
    