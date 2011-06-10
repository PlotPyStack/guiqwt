# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Mandelbrot demo"""

SHOW = True # Show test in GUI-based test launcher

import numpy as np

from guidata.qt.QtCore import QRectF, QPointF

from guiqwt.config import _
from guiqwt.plot import ImageDialog
from guiqwt.image import RawImageItem
from guiqwt.tools import ToggleTool

from guiqwt._mandel import mandel
mandelbrot = mandel.mandelbrot

class FullScale(ToggleTool):
    def __init__(self, parent, image):
        super(FullScale, self).__init__(parent, _(u"MAX resolution"), None)
        self.image = image
        self.minprec = image.IMAX
        self.maxprec = 5*image.IMAX
        
    def activate_command(self, plot, checked):
        if self.image.IMAX == self.minprec:
            self.image.IMAX = self.maxprec
        else:
            self.image.IMAX = self.minprec
        self.image.set_lut_range([0, self.image.IMAX])
        plot.replot()
        
    def update_status(self, plot):
        self.action.setChecked(self.image.IMAX == self.maxprec)

class MandelItem(RawImageItem):
    def __init__(self, xmin, xmax, ymin, ymax):
        super(MandelItem, self).__init__(np.zeros((1, 1), np.int16, order='F'))
        self.bounds = QRectF(QPointF(xmin, ymin),
                             QPointF(xmax, ymax))
        self.update_border()
        self.IMAX = 80
        self.set_lut_range([0, self.IMAX])
        
    #---- QwtPlotItem API ------------------------------------------------------
    def draw_image(self, painter, canvasRect, srcRect, dstRect, xMap, yMap):        
        x1, y1 = canvasRect.left(), canvasRect.top()
        x2, y2 = canvasRect.right(), canvasRect.bottom()
        i1, j1, i2, j2 = srcRect

        NX = x2-x1
        NY = y2-y1
        if self.data.shape != (NX, NY):
            self.data = np.zeros((NX, NY), np.int16, order='F')
            
        orig = complex(i1, j1)
        dx = (i2-i1)/(NX-1)
        dy = 1j*(j2-j1)/(NY-1)
        NMAX = self.IMAX
        mandelbrot(orig, dx, dy, self.data, NMAX)
        
        self.data = self.data.T
        
        srcRect = (0, 0, NX, NY)
        x1, y1, x2, y2 = canvasRect.getCoords()
        RawImageItem.draw_image(self, painter, canvasRect,
                                srcRect, (x1, y1, x2, y2), xMap, yMap)

def mandel():
    win = ImageDialog(edit=True, toolbar=True, wintitle="Mandelbrot",
                      options=dict(yreverse=False))
    mandel = MandelItem(-1.5, .5, -1., 1.)
    win.add_tool(FullScale, mandel)
    plot = win.get_plot()
    plot.set_aspect_ratio(lock=False)
    plot.add_item(mandel)
    plot.set_full_scale(mandel)
    win.show()
    win.exec_()

if __name__ == "__main__":
    import guidata
    _app = guidata.qapplication()
    mandel()
