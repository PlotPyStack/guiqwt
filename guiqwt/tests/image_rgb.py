# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut <pierre.raybaut@cea.fr>
# Licensed under the terms of the CECILL License
# (see guidata/__init__.py for details)

"""ImagePlotDialog test"""

import numpy as np

from guiqwt.plot import ImageDialog
from guiqwt.builder import make
import os.path as osp

SHOW = True # Show test in GUI-based test launcher

TESTDIR = osp.abspath(osp.dirname(__file__))
IMGFILE = osp.join(TESTDIR, "..", "images", "items", "image.png")

def imshow( data ):
    win = ImageDialog(edit=False, toolbar=True, wintitle="RGB image item test")
    item = make.rgb_image(data, scale=(-1,-1,1,1))
    plot = win.get_plot()
    plot.add_item(item)
    win.show()
    win.exec_()

def test():
    """Test"""
    # -- Create QApplication
    import guidata
    guidata.qapplication()
    # --
    from PIL.Image import open
    img = np.array(open(IMGFILE))
    print img.shape
    imshow(img)

if __name__ == "__main__":
    test()
