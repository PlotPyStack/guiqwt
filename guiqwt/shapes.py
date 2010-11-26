# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.shapes
-------------

The `shapes` module provides geometrical shapes:
    * :py:class:`guiqwt.shapes.PolygonShape`
    * :py:class:`guiqwt.shapes.RectangleShape`
    * :py:class:`guiqwt.shapes.PointShape`
    * :py:class:`guiqwt.shapes.SegmentShape`
    * :py:class:`guiqwt.shapes.EllipseShape`
    * :py:class:`guiqwt.shapes.Axes`
    * :py:class:`guiqwt.shapes.XRangeSelection`

A shape is a plot item (derived from QwtPlotItem) that may be displayed 
on a 2D plotting widget like :py:class:`guiqwt.curve.CurvePlot` 
or :py:class:`guiqwt.image.ImagePlot`.

.. seealso:: module :py:mod:`guiqwt.annotations`

Examples
~~~~~~~~

A shape may be created:
    * from the associated plot item class (e.g. `RectangleShape` to create a 
      rectangle): the item properties are then assigned by creating the 
      appropriate style parameters object
      (:py:class:`guiqwt.styles.ShapeParam`)
      
>>> from guiqwt.shapes import RectangleShape
>>> from guiqwt.styles import ShapeParam
>>> param = ShapeParam()
>>> param.title = 'My rectangle'
>>> rect_item = RectangleShape(0., 2., 4., 0., param)
      
    * or using the `plot item builder` (see :py:func:`guiqwt.builder.make`):
      
>>> from guiqwt.builder import make
>>> rect_item = make.rectangle(0., 2., 4., 0., title='My rectangle')

Reference
~~~~~~~~~

.. autoclass:: PolygonShape
   :members:
   :inherited-members:
.. autoclass:: RectangleShape
   :members:
   :inherited-members:
.. autoclass:: PointShape
   :members:
   :inherited-members:
.. autoclass:: SegmentShape
   :members:
   :inherited-members:
.. autoclass:: EllipseShape
   :members:
   :inherited-members:
.. autoclass:: Axes
   :members:
   :inherited-members:
.. autoclass:: XRangeSelection
   :members:
   :inherited-members:
"""

import sys, numpy as np
from math import fabs, sqrt, sin, cos, pi

from PyQt4.QtGui import QPen, QBrush, QPolygonF, QTransform, QPainter
from PyQt4.QtCore import Qt, QRectF, QPointF, QPoint, QLineF
from PyQt4.Qwt5 import QwtPlotItem, QwtSymbol, QwtPlotMarker

from guidata.utils import assert_interfaces_valid, update_dataset

# Local imports
from guiqwt.config import CONF, _
from guiqwt.interfaces import IBasePlotItem, IShapeItemType, ISerializableType
from guiqwt.styles import (MarkerParam, ShapeParam, RangeShapeParam,
                           AxesShapeParam)
from guiqwt.signals import (SIG_RANGE_CHANGED, SIG_MARKER_CHANGED,
                            SIG_AXES_CHANGED, SIG_ITEM_MOVED)

class AbstractShape(QwtPlotItem):
    """Interface pour les objets manipulables
    il n'est pas nécessaire de dériver de QwtShape si on
    réutilise une autre classe dérivée de QwtPlotItem
    
    La classe de base 
    """
    __implements__ = (IBasePlotItem,)

    _readonly = False
    _can_select = True
    _can_resize = True
    _can_rotate = False #TODO: implement shape rotation?
    _can_move = True
    
    def __init__(self):
        super(AbstractShape, self).__init__()
        self.selected = False
    
    def types(self):
        return (IShapeItemType, )
    def can_select(self):
        return self._can_select
    def can_resize(self):
        return self._can_resize
    def can_rotate(self):
        return self._can_rotate
    def can_move(self):
        return self._can_move

    def set_readonly(self, state):
        """Set object readonly state"""
        self._readonly = state
        
    def is_readonly(self):
        """Return object readonly state"""
        return self._readonly

    def hit_test(self, pos):
        """return (dist,handle,inside)"""
        pass
    
    def canvas_to_axes(self, pos):
        plot = self.plot()
        ax = self.xAxis()
        ay = self.yAxis()
        return plot.invTransform(ax, pos.x()), plot.invTransform(ay, pos.y())

    def axes_to_canvas(self, x, y):
        plot = self.plot()
        return plot.transform(self.xAxis(), x), plot.transform(self.yAxis(), y)

    def move_point_to(self, handle, pos):
        pass
    
    def move_local_point_to(self, handle, pos):
        pt = self.canvas_to_axes(pos)
        self.move_point_to(handle, pt)
        
    def move_local_shape(self, old_pos, new_pos):
        """Translate the shape such that old_pos becomes new_pos
        in canvas coordinates"""
        old_pt = self.canvas_to_axes(old_pos)
        new_pt = self.canvas_to_axes(new_pos)
        self.move_shape(old_pt, new_pt)
        if self.plot():
            self.plot().emit(SIG_ITEM_MOVED, self, *(old_pt+new_pt))

    def move_with_selection(self, dx, dy):
        """
        Translate the shape together with other selected items
        dx, dy: translation in plot coordinates
        """
        self.move_shape([0, 0], [dx, dy])

    def move_shape(self, old_pos, new_pos):
        """Translate the shape such that old_pos becomes new_pos
        in axis coordinates"""
        pass

    def invalidate_plot(self):
        plot = self.plot()
        if plot is not None:
            plot.invalidate()

    def select(self):
        """Select item"""
        self.selected = True
        self.invalidate_plot()
    
    def unselect(self):
        """Unselect item"""
        self.selected = False
        self.invalidate_plot()
        
    def get_item_parameters(self, itemparams):
        pass
    
    def set_item_parameters(self, itemparams):
        pass

assert_interfaces_valid(AbstractShape)


class Marker(QwtPlotMarker):
    """
    A marker that has two callbacks
    for restraining it's coordinates and
    displaying it's label
    we want to derive from QwtPlotMarker so
    we have to reimplement some of AbstractShape's methods
    (and PyQt doesn't really like multiple inheritance...)
    """
    __implements__ = (IBasePlotItem,)
    _readonly = True

    def __init__(self, label_cb=None, constraint_cb=None):
        super(Marker, self).__init__()
        self.selected = False
        self.label_cb = label_cb
        self.constraint_cb = constraint_cb
        self.markerparam = MarkerParam(_("Marker"))
        self.markerparam.update_marker(self)

    def set_style(self, section, option):
        self.markerparam.read_config(CONF, section, option)
        self.markerparam.update_marker(self)

    def types(self):
        return (IShapeItemType,)
    def can_select(self):
        return True
    def can_resize(self):
        return True
    def can_rotate(self):
        return False
    def can_move(self):
        return True

    def set_readonly(self, state):
        """Set object readonly state"""
        self._readonly = state
        
    def is_readonly(self):
        """Return object readonly state"""
        return self._readonly

    def hit_test(self, pos):
        """return (dist,handle,inside)"""
        plot = self.plot()
        xc, yc = pos.x(), pos.y()
        x = plot.transform(self.xAxis(), self.xValue())
        y = plot.transform(self.yAxis(), self.yValue())
        ls = self.markerparam.linestyle
        if ls == 0:
            return sqrt((x-xc)**2 + (y-yc)**2), 0, False, None
        elif ls == 1:
            return sqrt((y-yc)**2), 0, False, None
        elif ls == 2:
            return sqrt((x-xc)**2), 0, False, None
        elif ls == 3:
            return sqrt(min((x-xc)**2,(y-yc)**2) ), 0, False, None
            
    
    def canvas_to_axes(self, pos):
        plot = self.plot()
        if plot is None:
            return
        ax = self.xAxis()
        ay = self.yAxis()
        return plot.invTransform(ax, pos.x()), plot.invTransform(ay, pos.y())

    def axes_to_canvas(self, x, y):
        plot = self.plot()
        return plot.transform(self.xAxis(), x), plot.transform(self.yAxis(), y)

    def move_point_to(self, handle, pos):
        x, y = pos
        if self.constraint_cb:
            x, y = self.constraint_cb(self, x, y)
        self.setValue(x, y)
        self.update_label()
        if self.plot():
            self.plot().emit(SIG_MARKER_CHANGED, self)
    
    def move_local_point_to(self, handle, pos):
        pt = self.canvas_to_axes(pos)
        self.move_point_to(handle, pt)
        
    def move_local_shape(self, old_pos, new_pos):
        """Translate the shape such that old_pos becomes new_pos
        in canvas coordinates"""
        old_pt = self.canvas_to_axes(old_pos)
        new_pt = self.canvas_to_axes(new_pos)
        self.move_shape(old_pt, new_pt)

    def move_with_selection(self, dx, dy):
        """
        Translate the shape together with other selected items
        dx, dy: translation in plot coordinates
        """
        self.move_shape([0, 0], [dx, dy])

    def move_shape(self, old_pos, new_pos):
        """Translate the shape such that old_pos becomes new_pos
        in canvas coordinates"""
        dx = new_pos[0]-old_pos[0]
        dy = new_pos[1]-old_pos[1]
        x, y = self.xValue(), self.yValue()
        return self.move_point_to(0, (x+dx, y+dy))

    def invalidate_plot(self):
        plot = self.plot()
        if plot is not None:
            plot.invalidate()

    def select(self):
        if self.selected:
            # Already selected
            return
        self.selected = True
        symb = self.symbol()
        pen = symb.pen()
        pen.setWidth(2)
        symb.setPen(pen)
        symb = QwtSymbol(symb.style(), symb.brush(), pen, symb.size())
        self.setSymbol(symb)
        self.invalidate_plot()

    def unselect(self):
        self.selected = False
        self.markerparam.update_marker(self)
        self.invalidate_plot()
        
    def update_label(self):
        x = self.xValue()
        y = self.yValue()
        if self.label_cb:
            label = self.label_cb(self, x, y)
            if label is None:
                return
        else:
            label = "x = %f<br>y = %f" % (x, y)
        text = self.label()
        text.setText(label)
        self.setLabel(text)
        xaxis = self.plot().axisScaleDiv(self.xAxis())
        if x < (xaxis.upperBound()+xaxis.lowerBound())/2:
            hor_alignment = Qt.AlignRight
        else:
            hor_alignment = Qt.AlignLeft
        yaxis = self.plot().axisScaleDiv(self.yAxis())
        ymap = self.plot().canvasMap(self.yAxis())
        y_top, y_bottom = ymap.s1(), ymap.s2()
        if y < (yaxis.upperBound()+yaxis.lowerBound())/2:
            if y_top > y_bottom:
                ver_alignment = Qt.AlignBottom
            else:
                ver_alignment = Qt.AlignTop
        else:
            if y_top > y_bottom:
                ver_alignment = Qt.AlignTop
            else:
                ver_alignment = Qt.AlignBottom
        self.setLabelAlignment(hor_alignment | ver_alignment)
        
    def get_item_parameters(self, itemparams):
        self.markerparam.update_param(self)
        itemparams.add("MarkerParam", self, self.markerparam)
    
    def set_item_parameters(self, itemparams):
        update_dataset(self.markerparam, itemparams.get("MarkerParam"),
                       visible_only=True)
        self.markerparam.update_marker(self)
        if self.selected:
            self.select()

assert_interfaces_valid(Marker)


class PolygonShape(AbstractShape):
    def __init__(self, points, closed=True):
        super(PolygonShape, self).__init__()
        self.closed = closed
        self.selected = False
        
        self.shapeparam = ShapeParam(_("Shape"), icon="rectangle.png")
        
        self.pen = QPen()
        self.brush = QBrush()
        self.symbol = QwtSymbol.NoSymbol
        self.sel_pen = QPen()
        self.sel_brush = QBrush()
        self.sel_symbol = QwtSymbol.NoSymbol
        self.points = np.zeros( (0, 2), float )
        if points:
            self.set_points(points)
                
    def types(self):
        return (IShapeItemType, ISerializableType)

    def __reduce__(self):
        state = (self.shapeparam, self.points, self.z())
        return (PolygonShape, (None, self.closed), state)

    def __setstate__(self, state):
        param, points, z = state
        self.points = points
        self.setZ(z)
        self.shapeparam = param
        self.shapeparam.update_shape(self)

    def set_style(self, section, option):
        self.shapeparam.read_config(CONF, section, option)
        self.shapeparam.update_shape(self)

    def set_points(self, points):
        self.points = np.array(points, float)
        assert self.points.shape[1] == 2
        
    def get_points(self):
        """Return polygon points"""
        return self.points
        
    def transform_points(self, xMap, yMap):
        points = QPolygonF()
        for i in xrange(self.points.shape[0]):
            points.append(QPointF(xMap.transform(self.points[i, 0]),
                                  yMap.transform(self.points[i, 1])))
        return points
    
    def get_reference_point(self):
        return self.points.mean(axis=0)

    def get_pen_brush(self, xMap, yMap):
        if self.selected:
            pen = self.sel_pen
            brush = self.sel_brush
            sym = self.sel_symbol
        else:
            pen = self.pen
            brush = self.brush
            sym = self.symbol

        tr = brush.transform()
        x0, y0 = self.get_reference_point()
        xx0 = xMap.transform(x0)
        yy0 = yMap.transform(y0)
        t0 = QTransform.fromTranslate(xx0, yy0)
        tr = tr*t0
        brush = QBrush(brush)
        brush.setTransform(tr)
        return pen, brush, sym

    def draw(self, painter, xMap, yMap, canvasRect):
        pen, brush, symbol = self.get_pen_brush(xMap, yMap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pen)
        painter.setBrush(brush)
        points = self.transform_points(xMap, yMap)
        if self.closed:
            painter.drawPolygon(points)
        else:
            painter.drawPolyline(points)
        if symbol != QwtSymbol.NoSymbol:
            for i in xrange(points.size()):
                symbol.draw(painter, points[i].toPoint())
    
    def poly_hit_test(self, plot, ax, ay, pos):
        pos = QPointF(pos)
        dist = sys.maxint
        handle = -1
        Cx, Cy = pos.x(), pos.y()
        poly = QPolygonF()
        pts = self.points
        for i in xrange(pts.shape[0]):
            # On calcule la distance dans le repère du canvas
            px = plot.transform(ax, pts[i, 0])
            py = plot.transform(ay, pts[i, 1])
            poly.append(QPointF(px, py))
            d = (Cx-px)**2 + (Cy-py)**2
            if d < dist:
                dist = d
                handle = i
        inside = poly.containsPoint(QPointF(Cx, Cy), Qt.OddEvenFill)
        return sqrt(dist), handle, inside, None

    def hit_test(self, pos):
        """return (dist, handle, inside)"""
        if not self.plot():
            return sys.maxint, 0, False, None
        return self.poly_hit_test(self.plot(), self.xAxis(), self.yAxis(), pos)
    
    def add_local_point(self, pos):
        pt = self.canvas_to_axes(pos)
        return self.add_point(pt)
        
    def add_point(self, pt):
        N,_ = self.points.shape
        self.points = np.resize(self.points, (N+1, 2))
        self.points[N, :] = pt
        return N

    def del_point(self, handle):
        self.points = np.delete(self.points, handle, 0)
        if handle < len(self.points):
            return handle
        else:
            return self.points.shape[0]-1
    
    def move_point_to(self, handle, pos):
        self.points[handle, :] = pos
        
    def move_shape(self, old_pos, new_pos):
        dx = new_pos[0]-old_pos[0]
        dy = new_pos[1]-old_pos[1]
        self.points += np.array([[dx, dy]])

    def get_item_parameters(self, itemparams):
        self.shapeparam.update_param(self)
        itemparams.add("ShapeParam", self, self.shapeparam)
    
    def set_item_parameters(self, itemparams):
        update_dataset(self.shapeparam, itemparams.get("ShapeParam"),
                       visible_only=True)
        self.shapeparam.update_shape(self)
        
assert_interfaces_valid(PolygonShape)


class PointShape(PolygonShape):
    def __init__(self, x, y):
        super(PointShape, self).__init__([], closed=False)
        self.set_pos(x, y)
        
    def set_pos(self, x, y):
        """Set the point coordinates to (x, y)"""
        self.set_points([(x, y)])
        
    def get_pos(self):
        """Return the point coordinates"""
        return tuple(self.points[0])
    
    def move_point_to(self, handle, pos):
        nx, ny = pos
        self.points[0] = (nx, ny)

    def __reduce__(self):
        state = (self.shapeparam, self.points, self.z())
        return (self.__class__, (0,0), state)

assert_interfaces_valid(PointShape)


class SegmentShape(PolygonShape):
    def __init__(self, x1, y1, x2, y2):
        super(SegmentShape, self).__init__([], closed=False)
        self.set_rect(x1, y1, x2, y2)
        
    def set_rect(self, x1, y1, x2, y2):
        """
        Set the start point of this segment to (x1, y1) 
        and the end point of this line to (x2, y2)
        """
        self.set_points([(x1, y1), (0, 0), (x2, y2)])

    def get_rect(self):
        return tuple(self.points[0])+tuple(self.points[2])

    def draw(self, painter, xMap, yMap, canvasRect):
        self.points[1] = (self.points[0]+self.points[2])/2.
        super(SegmentShape, self).draw(painter, xMap, yMap, canvasRect)
    
    def move_point_to(self, handle, pos):
        nx, ny = pos
        if handle == 0:
            self.points[0] = (nx, ny)
        elif handle == 2:
            self.points[2] = (nx, ny)
        elif handle in (1, -1):
            delta = (nx, ny)-self.points.mean(axis=0)
            self.points += delta

    def __reduce__(self):
        state = (self.shapeparam, self.points, self.z())
        return (self.__class__, (0,0,0,0), state)

assert_interfaces_valid(SegmentShape)


class RectangleShape(PolygonShape):
    def __init__(self, x1, y1, x2, y2):
        super(RectangleShape, self).__init__([], closed=True)
        self.set_rect(x1, y1, x2, y2)
        
    def set_rect(self, x1, y1, x2, y2):
        """
        Set the coordinates of the rectangle's top-left corner to (x1, y1), 
        and of its bottom-right corner to (x2, y2).
        """
        self.set_points([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def get_rect(self):
        return tuple(self.points[0])+tuple(self.points[2])

    def move_point_to(self, handle, pos):
        nx, ny = pos
        x1, y1 = self.points[0]
        x2, y2 = self.points[2]
        if handle == 0:
            self.set_rect(nx, ny, x2, y2)
        elif handle == 1:
            self.set_rect(x1, ny, nx, y2)
        elif handle == 2:
            self.set_rect(x1, y1, nx, ny)
        elif handle == 3:
            self.set_rect(nx, y1, x2, ny)
        elif handle == -1:
            delta = (nx, ny)-self.points.mean(axis=0)
            self.points += delta

    def __reduce__(self):
        state = (self.shapeparam, self.points, self.z())
        return (self.__class__, (0,0,0,0), state)

assert_interfaces_valid(RectangleShape)


#FIXME: EllipseShape's ellipse drawing is invalid when aspect_ratio != 1
class EllipseShape(RectangleShape):
    def __init__(self, x1, y1, x2, y2, ratio=None):
        self.is_ellipse = False
        self.ratio = ratio
        super(EllipseShape, self).__init__(x1, y1, x2, y2)
        
    def switch_to_ellipse(self):
        self.is_ellipse = True

    def set_rect(self, x1, y1, x2, y2):
        """
        Set the start point of the ellipse's X-axis diameter to (x1, y1) 
        and its end point to (x2, y2)
        """
        self.set_xdiameter(x1, y1, x2, y2)

    def compute_elements(self, xMap, yMap):
        """Return points, lines and ellipse rect"""
        points = self.transform_points(xMap, yMap)
        line0 = QLineF(points[0], points[1])
        line1 = QLineF(points[2], points[3])        
        rect = QRectF()
        rect.setWidth(line0.length())
        rect.setHeight(line1.length())
        rect.moveCenter(line0.pointAt(.5))
        return points, line0, line1, rect

    def hit_test(self, pos):
        """return (dist, handle, inside)"""
        if not self.plot():
            return sys.maxint, 0, False, None
        dist, handle, inside, other = self.poly_hit_test(self.plot(),
                                             self.xAxis(), self.yAxis(), pos)
        if not inside:
            xMap = self.plot().canvasMap(self.xAxis())
            yMap = self.plot().canvasMap(self.yAxis())
            _points, _line0, _line1, rect = self.compute_elements(xMap, yMap)
            inside = rect.contains(QPointF(pos))
        return dist, handle, inside, other

    def draw(self, painter, xMap, yMap, canvasRect):
        points, line0, line1, rect = self.compute_elements(xMap, yMap)
        pen, brush, symbol = self.get_pen_brush(xMap, yMap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawLine(line0)
        painter.drawLine(line1)
        painter.save()
        painter.translate(rect.center())
        painter.rotate(-line0.angle())
        painter.translate(-rect.center())
        painter.drawEllipse(rect.toRect())
        painter.restore()
        if symbol != QwtSymbol.NoSymbol:
            for i in xrange(points.size()):
                symbol.draw(painter, points[i].toPoint())

    def get_xline(self):
        return QLineF(*(tuple(self.points[0])+tuple(self.points[1])))

    def get_yline(self):
        return QLineF(*(tuple(self.points[2])+tuple(self.points[3])))

    def set_xdiameter(self, x0, y0, x1, y1):
        xline = QLineF(x0, y0, x1, y1)
        yline = xline.normalVector()
        yline.translate(xline.pointAt(.5)-xline.p1())
        if self.is_ellipse:
            yline.setLength(self.get_yline().length())
        elif self.ratio is not None:
            yline.setLength(xline.length()*self.ratio)
        yline.translate(yline.pointAt(.5)-yline.p2())
        self.set_points([(x0, y0), (x1, y1),
                         (yline.x1(), yline.y1()), (yline.x2(), yline.y2())])
                         
    def set_ydiameter(self, x2, y2, x3, y3):
        yline = QLineF(x2, y2, x3, y3)
        xline = yline.normalVector()
        xline.translate(yline.pointAt(.5)-yline.p1())
        if self.is_ellipse:
            xline.setLength(self.get_xline().length())
        xline.translate(xline.pointAt(.5)-xline.p2())
        self.set_points([(xline.x1(), xline.y1()), (xline.x2(), xline.y2()),
                         (x2, y2), (x3, y3)])

    def move_point_to(self, handle, pos):
        nx, ny = pos
        if handle == 0:
            x1, y1 = self.points[1]
            self.set_xdiameter(nx, ny, x1, y1)
        elif handle == 1:
            x0, y0 = self.points[0]
            self.set_xdiameter(x0, y0, nx, ny)
        elif handle == 2 and self.is_ellipse:
            x3, y3 = self.points[3]
            self.set_ydiameter(nx, ny, x3, y3)
        elif handle == 3 and self.is_ellipse:
            x2, y2 = self.points[2]
            self.set_ydiameter(x2, y2, nx, ny)
        elif handle in (2, 3):
            delta = (nx, ny)-self.points[handle]
            self.points += delta
        elif handle == -1:
            delta = (nx, ny)-self.points.mean(axis=0)
            self.points += delta

assert_interfaces_valid(EllipseShape)


class Axes(PolygonShape):
    """Axes( (0,1), (1,1), (0,0) )"""
    def __init__(self, p0, p1, p2):
        super(Axes, self).__init__([], closed=True)
        self.set_rect(p0, p1, p2)
        self.arrow_angle = 15 # degrees
        self.arrow_size = 0.05 # % of axe length
        self.x_pen = self.pen
        self.x_brush = self.brush
        self.y_pen = self.pen
        self.y_brush = self.brush
        self.axesparam = AxesShapeParam(_("Axes"), icon="gtaxes.png")
        self.axesparam.update_param(self)

    def __reduce__(self):
        state = (self.shapeparam, self.axesparam, self.points, self.z())
        return (self.__class__, ((0, 0), (0, 0), (0, 0)), state)

    def __setstate__(self, state):
        shapeparam, axesparam, points, z = state
        self.points = points
        self.setZ(z)
        self.shapeparam = shapeparam
        self.shapeparam.update_shape(self)
        self.axesparam = axesparam
        self.axesparam.update_axes(self)
        
    def get_transform_matrix(self, dx=1., dy=1.):
        p0, p1, _p3, p2 = [np.array([p[0], p[1], 1.0]) for p in self.points]
        matrix = np.array([(p1-p0)/dx, (p2-p0)/dy, p0])
        if abs(np.linalg.det(matrix)) > 1e-12:
            return np.linalg.inv(matrix)

    def set_rect(self, p0, p1, p2):
        p3x = p1[0]+p2[0]-p0[0]
        p3y = p1[1]+p2[1]-p0[1]
        self.set_points([p0, p1, (p3x, p3y), p2])

    def set_style(self, section, option):
        super(Axes, self).set_style(section, option+"/border")
        self.axesparam.read_config(CONF, section, option)
        self.axesparam.update_axes(self)

    def move_point_to(self, handle, pos):
        _nx, _ny = pos
        p0, p1, _p3, p2 = list(self.points)
        d1x = p1[0]-p0[0]
        d1y = p1[1]-p0[1]
        d2x = p2[0]-p0[0]
        d2y = p2[1]-p0[1]
        if handle == 0:
            pp0 = pos
            pp1 = pos[0] + d1x, pos[1] + d1y
            pp2 = pos[0] + d2x, pos[1] + d2y
        elif handle == 1:
            pp0 = p0
            pp1 = pos
            pp2 = p2
        elif handle == 3:
            pp0 = p0
            pp1 = p1
            pp2 = pos
        elif handle == 2:
            # find (a,b) such that p3 = a*d1 + b*d2 + p0
            d3x = pos[0]-p0[0]
            d3y = pos[1]-p0[1]
            det = d1x*d2y-d2x*d1y
            if abs(det)<1e-6:
                # reset
                d1x = d2y = 1.
                d1y = d2x = 0.
                det = 1.
            a = (d2y*d3x - d2x*d3y) / det
            b = (-d1y*d3x + d1x*d3y) / det
            _pp3 = pos
            pp1 = p0[0] + a*d1x, p0[1] + a*d1y
            pp2 = p0[0] + b*d2x, p0[1] + b*d2y
            pp0 = p0
        self.set_rect(pp0, pp1, pp2)
        if self.plot():
            self.plot().emit(SIG_AXES_CHANGED, self)
            
    def move_shape(self, old_pos, new_pos):
        """Overriden to emit the axes_changed signal"""
        PolygonShape.move_shape(self, old_pos, new_pos)
        if self.plot():
            self.plot().emit(SIG_AXES_CHANGED, self)

    def draw(self, painter, xMap, yMap, canvasRect):
        super(Axes, self).draw(painter, xMap, yMap, canvasRect)
        p0, p1, _, p2 = list(self.points)

        painter.setPen(self.x_pen)
        painter.setBrush(self.x_brush)
        self.draw_arrow(painter, xMap, yMap, p0, p1)
        painter.setPen(self.y_pen)
        painter.setBrush(self.y_brush)
        self.draw_arrow(painter, xMap, yMap, p0, p2)
        
    def draw_arrow(self, painter, xMap, yMap, p0, p1):
        sz = self.arrow_size
        angle = pi*self.arrow_angle/180.
        ca, sa = cos(angle), sin(angle)
        d1x = (xMap.transform(p1[0])-xMap.transform(p0[0]))
        d1y = (yMap.transform(p1[1])-yMap.transform(p0[1]))
        norm = sqrt(d1x**2+d1y**2)
        if abs(norm) < 1e-6:
            return
        d1x *= sz/norm
        d1y *= sz/norm
        n1x = -d1y
        n1y = d1x
        # arrow : a0 - a1 == p1 - a2
        a1x = xMap.transform(p1[0])
        a1y = yMap.transform(p1[1])
        a0x = a1x - ca*d1x + sa*n1x
        a0y = a1y - ca*d1y + sa*n1y
        a2x = a1x - ca*d1x - sa*n1x
        a2y = a1y - ca*d1y - sa*n1y
        
        poly = QPolygonF()
        poly.append(QPointF(a0x, a0y))
        poly.append(QPointF(a1x, a1y))
        poly.append(QPointF(a2x, a2y))
        painter.drawPolygon(poly)
        
    def get_item_parameters(self, itemparams):
        super(Axes, self).get_item_parameters(itemparams)
        self.axesparam.update_param(self)
        itemparams.add("AxesShapeParam", self, self.axesparam)
    
    def set_item_parameters(self, itemparams):
        super(Axes, self).set_item_parameters(itemparams)
        update_dataset(self.axesparam, itemparams.get("AxesShapeParam"),
                       visible_only=True)
        self.axesparam.update_axes(self)
        
assert_interfaces_valid(Axes)

        
class XRangeSelection(AbstractShape):
    def __init__(self, _min, _max):
        super(XRangeSelection,self).__init__()
        self._min = _min
        self._max = _max
        self.shapeparam = RangeShapeParam(_("Range"), icon="xrange.png")
        self.shapeparam.read_config(CONF, "histogram", "range")
        self.pen = None
        self.sel_pen = None
        self.brush = None
        self.handle = None
        self.symbol = None
        self.sel_symbol = None
        self.shapeparam.update_range(self) # creates all the above QObjects
        
    def get_handles_pos(self):
        plot = self.plot()
        rct = plot.canvas().contentsRect()
        y = rct.center().y()
        x0 = plot.transform(self.xAxis(), self._min)
        x1 = plot.transform(self.xAxis(), self._max)
        return x0, x1, y
        
    def draw(self, painter, xMap, yMap, canvasRect):
        plot = self.plot()
        if not plot:
            return
        if self.selected:
            pen = self.sel_pen
            sym = self.sel_symbol
        else:
            pen = self.pen
            sym = self.symbol
            
        rct = plot.canvas().contentsRect()
        rct2 = QRectF(rct)
        rct2.setLeft(xMap.transform(self._min))
        rct2.setRight(xMap.transform(self._max))
        
        painter.fillRect(rct2, self.brush)
        painter.setPen(pen)
        painter.drawLine(rct2.topLeft(), rct2.bottomLeft())
        painter.drawLine(rct2.topRight(), rct2.bottomRight())
        dash = QPen(pen)
        dash.setStyle(Qt.DashLine)
        dash.setWidth(1)
        painter.setPen(dash)
        painter.drawLine(rct2.center().x(), rct2.top(),
                         rct2.center().x(), rct2.bottom())
        painter.setPen(pen)
        x0, x1, y = self.get_handles_pos()        
        sym.draw(painter, QPoint(x0,y))
        sym.draw(painter, QPoint(x1,y))
        
    def get_handle_rect(self, xMap, yMap, xpos, ypos):
        cx = xMap.transform(xpos)
        cy = yMap.transform(ypos)
        hs = self._handle_size
        return QRectF(cx-hs, cy-hs, 2*hs, 2*hs)
        
    def hit_test(self, pos):
        x, _y = pos.x(), pos.y()
        x0, x1, _yp = self.get_handles_pos()
        d0 = fabs(x0-x)
        d1 = fabs(x1-x)
        d2 = fabs((x0+x1)/2-x)
        z = np.array([d0, d1, d2])
        dist = z.min()
        handle = z.argmin()
        inside = bool(x0<x<x1)
        return dist, handle, inside, None
        
    def move_local_point_to(self, handle, pos):
        val = self.plot().invTransform(self.xAxis(), pos.x())
        self.move_point_to(handle, (val, 0))
        
    def move_point_to(self, hnd, pos):
        val, _ = pos
        if hnd == 0:
            self._min = val
        elif hnd == 1:
            self._max = val
        elif hnd == 2:
            move = val-(self._max+self._min)/2
            self._min += move
            self._max += move

        self.plot().emit(SIG_RANGE_CHANGED, self, self._min, self._max)
        #self.plot().replot()

    def get_range(self):
        return self._min, self._max

    def set_range(self, _min, _max, dosignal=True):
        self._min = _min
        self._max = _max
        if dosignal:
            self.plot().emit(SIG_RANGE_CHANGED, self, self._min, self._max)

    def move_shape(self, old_pos, new_pos):
        dx = new_pos[0]-old_pos[0]
        _dy = new_pos[1]-old_pos[1]
        self._min += dx
        self._max += dx
        self.plot().emit(SIG_RANGE_CHANGED, self, self._min, self._max)
        self.plot().replot()
        
    def get_item_parameters(self, itemparams):
        self.shapeparam.update_param(self)
        itemparams.add("ShapeParam", self, self.shapeparam)
    
    def set_item_parameters(self, itemparams):
        update_dataset(self.shapeparam, itemparams.get("ShapeParam"),
                       visible_only=True)
        self.shapeparam.update_range(self)
        self.sel_brush = QBrush(self.brush)
        
assert_interfaces_valid(XRangeSelection)