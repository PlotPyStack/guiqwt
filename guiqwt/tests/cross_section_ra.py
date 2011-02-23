# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Radially-averaged cross section test"""

SHOW = True # Show test in GUI-based test launcher

import numpy as np

from guiqwt.plot import ImageDialog
from guiqwt.builder import make
from guiqwt.cross_section import RACrossSection
from guiqwt.tools import RACrossSectionTool, RACSPanelTool, ImageMaskTool

class RACSImageDialog(ImageDialog):
    def register_image_tools(self):
        ImageDialog.register_image_tools(self)
        for tool in (RACrossSectionTool, RACSPanelTool, ImageMaskTool):
            self.add_tool(tool)
        
    def create_plot(self, options, row=0, column=0, rowspan=1, columnspan=1):
        ImageDialog.create_plot(self, options, row, column, rowspan, columnspan)
        ra_panel = RACrossSection(self)
        splitter = self.plot_widget.xcsw_splitter
        splitter.addWidget(ra_panel)
        splitter.setStretchFactor(splitter.count()-1, 1)
        splitter.setSizes(list(splitter.sizes())+[2])
        self.add_panel(ra_panel)

def test():
    """Test"""
    # -- Create QApplication
    import guidata
    guidata.qapplication()
    # --
    win = RACSImageDialog(toolbar=True,
                          wintitle="Radially-averaged cross section test")
    win.resize(600, 600)
    
    from guiqwt.tests.image import compute_image
    data = compute_image(4000, grid=False)
    data = np.uint16((data+1)*32767)
    data = data[:2000, :2000]
    print data.dtype

    image = make.maskedimage(data, colormap="bone", show_mask=True)
    plot = win.get_plot()
    plot.add_item(image)
    win.exec_()
    
def benchmark():
    """Fortran vs. Python/NumPy comparative benchmark"""
    from guiqwt._ext import radialaverage
    fortran_func = radialaverage.radavg
    from guiqwt.cross_section import radial_average as python_func
    
    from guiqwt.tests.image import compute_image
    N = 1000
    data = compute_image(N, grid=False)
    
    ix0, iy0 = 100, 100
    ix1, iy1 = N-100, N-100
    ixc, iyc = N/2, N/2
    iradius = (N-100*2)/2
    
    import time
    t0 = time.time()
    y_python = python_func(data, ix0, iy0, ix1, iy1, ixc, iyc, iradius)
    t1 = time.time()
    print "Python/NumPy: %05d ms" % round((t1-t0)*1e3)

    y_fortran = np.zeros((iradius+1,), dtype=np.float64)
    ycount = np.zeros((iradius+1,), dtype=np.float64)
    fortran_func(y_fortran, ycount, data, iyc, ixc, iradius)
    t2 = time.time()
    print "     Fortran: %05d ms" % round((t2-t1)*1e3)
    
    import guiqwt.pyplot as plt
    plt.figure("Test image")
    plt.imshow(data)
    plt.figure("Radially-averaged cross section")
    plt.plot(y_python)
    plt.plot(y_fortran)
    plt.show()

if __name__ == "__main__":
    test()
#    benchmark()
