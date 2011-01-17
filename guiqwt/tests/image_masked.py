# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guidata/__init__.py for details)

"""
Masked Image test, creating the MaskedImageItem object via make.maskedimage

Masked image items are constructed using a masked array item. Masked data is 
ignored in computations, like the average cross sections.
"""

SHOW = True # Show test in GUI-based test launcher

import os.path as osp, numpy as np

from guiqwt.plot import ImageDialog
from guiqwt.io import imagefile_to_array
from guiqwt.builder import make

SHOW = True # Show test in GUI-based test launcher

if __name__ == "__main__":
    import guidata
    guidata.qapplication()
    win = ImageDialog(toolbar=True, wintitle="Masked image item test")
    data = imagefile_to_array(osp.join(osp.abspath(osp.dirname(__file__)),
                                       "brain.png"))
    mask = np.zeros_like(data)
    mask[20:120, 20:120] = True
    print mask
    win.get_plot().add_item( make.maskedimage(data, mask, colormap='gray',
                                              show_mask=True) )
    win.show()
    win.exec_()
