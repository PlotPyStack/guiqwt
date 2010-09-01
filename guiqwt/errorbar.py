# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
Errorbar curve object
"""

from PyQt4.Qwt5 import QwtPlotCurve, QwtScaleMap
from PyQt4.QtCore import QPoint, QRectF, QLine
from PyQt4.QtGui import QBrush

from numpy import array, vectorize

from guidata.utils import assert_interfaces_valid, update_dataset

# Local imports
from guiqwt.styles import ErrorBarParam
from guiqwt.curve import CurveItem


def _transform(map,v):
    return QwtScaleMap.transform(map,v)
vmap = vectorize(_transform)


class ErrorBarCurveItem(CurveItem):
    """ErrorBar curve"""
    def __init__(self, curveparam=None, errorbarparam=None):
        if errorbarparam is None:
            self.errorbarparam = ErrorBarParam()
        else:
            self.errorbarparam = errorbarparam
        super(ErrorBarCurveItem, self).__init__(curveparam)
        self._dx = None
        self._dy = None
        
    def unselect(self):
        super(ErrorBarCurveItem, self).unselect()
        self.errorbarparam.update_curve(self)

    def set_data(self, x, y, dx, dy):
        CurveItem.set_data(self, x, y)
        if dx is not None:
            dx = array(dx, copy=False)
            if dx.size == 0:
                dx = None
        if dy is not None:
            dy = array(dy, copy=False)
            if dy.size == 0:
                dy = None
        self._dx = dx
        self._dy = dy

    def get_minmax_arrays(self):
        x = self._x
        y = self._y
        dx = self._dx
        dy = self._dy
        if dx is None:
            xmin = x
            xmax = x
        else:
            xmin = x - dx
            xmax = x + dx
        if dy is None:
            ymin = y
            ymax = y
        else:
            ymin = y - dy
            ymax = y + dy
        return xmin, xmax, ymin, ymax
        
    def get_closest_coordinates(self, x, y):
        # Surcharge d'une méthode de base de CurveItem
        plot = self.plot()
        ax = self.xAxis()
        ay = self.yAxis()
        xc = plot.transform(ax, x)
        yc = plot.transform(ay, y)
        _distance, i, _inside, _other = self.hit_test(QPoint(xc, yc))
        x0, y0 = self.plot().canvas2plotitem(self, xc, yc)
        x = self.x(i)
        y = self.y(i)
        xmin, xmax, ymin, ymax = self.get_minmax_arrays()
        if abs(y0-y) > abs(y0-ymin[i]):
            y = ymin[i]
        elif abs(y0-y) > abs(y0-ymax[i]):
            y = ymax[i]
        if abs(x0-x) > abs(x0-xmin[i]):
            x = xmin[i]
        elif abs(x0-x) > abs(x0-xmax[i]):
            x = xmax[i]
        return x, y

    def boundingRect(self):
        """Return the bounding rectangle of the data, error bars included"""
        xmin, xmax, ymin, ymax = self.get_minmax_arrays()
        return QRectF( xmin.min(), ymin.min(),
                       xmax.max()-xmin.min(), ymax.max()-ymin.min() )
        
    def draw(self, painter, xMap, yMap, canvasRect):
        """Draw an interval of the curve, including the error bars
        painter is the QPainter used to draw the curve
        xMap is the Qwt.QwtDiMap used to map x-values to pixels
        yMap is the Qwt.QwtDiMap used to map y-values to pixels
        first is the index of the first data point to draw
        last is the index of the last data point to draw. If last < 0, last
        is transformed to index the last data point
        """
        x = self._x
        y = self._y
        tx = vmap(xMap, x)
        ty = vmap(yMap, y)
        xmin, xmax, ymin, ymax = self.get_minmax_arrays()
        RN = xrange(len(tx))
        if self.errorOnTop:
            QwtPlotCurve.draw(self, painter, xMap, yMap, canvasRect)
        
        painter.save()
        painter.setPen(self.errorPen)
        cap = self.errorCap/2.

        if self._dx is not None and self.errorbarparam.mode == 0:
            txmin = vmap(xMap, xmin)
            txmax = vmap(xMap, xmax)
            # Classic error bars
            lines = []
            for i in RN:
                yi = ty[i]
                lines.append(QLine(txmin[i], yi, txmax[i], yi))
            painter.drawLines(lines)
            if cap > 0:
                lines = []
                for i in RN:
                    yi = ty[i]
                    lines.append(QLine(txmin[i], yi-cap, txmin[i], yi+cap))
                    lines.append(QLine(txmax[i], yi-cap, txmax[i], yi+cap))
            painter.drawLines(lines)
            
        if self._dy is not None:
            tymin = vmap(yMap, ymin)
            tymax = vmap(yMap, ymax)
            if self.errorbarparam.mode == 0:
                # Classic error bars
                lines = []
                for i in RN:
                    xi = tx[i]
                    lines.append(QLine(xi, tymin[i], xi, tymax[i]))
                painter.drawLines(lines)
                if cap > 0:
                    # Cap
                    lines = []
                    for i in RN:
                        xi = tx[i]
                        lines.append(QLine(xi-cap, tymin[i], xi+cap, tymin[i]))
                        lines.append(QLine(xi-cap, tymax[i], xi+cap, tymax[i]))
                painter.drawLines(lines)
            else:
                # Error area
                points = []
                rpoints = []
                for i in RN:
                    xi = tx[i]
                    points.append(QPoint(xi, tymin[i]))
                    rpoints.append(QPoint(xi, tymax[i]))
                points+=reversed(rpoints)
                painter.setBrush(QBrush(self.errorBrush))
                painter.drawPolygon(*points)

        painter.restore()

        if not self.errorOnTop:
            QwtPlotCurve.draw(self, painter, xMap, yMap, canvasRect)
        
    def update_params(self):
        self.errorbarparam.update_curve(self)
        super(ErrorBarCurveItem, self).update_params()

    def get_item_parameters(self, itemparams):
        super(ErrorBarCurveItem, self).get_item_parameters(itemparams)
        itemparams.add("ErrorBarParam", self, self.errorbarparam)
    
    def set_item_parameters(self, itemparams):
        update_dataset(self.errorbarparam, itemparams.get("ErrorBarParam"),
                       visible_only=True)
        super(ErrorBarCurveItem, self).set_item_parameters(itemparams)

assert_interfaces_valid( ErrorBarCurveItem )
