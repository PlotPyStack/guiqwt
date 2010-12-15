# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.curve
------------

The `curve` module provides curve-related objects:
    * :py:class:`guiqwt.curve.CurvePlot`: a 2d curve plotting widget
    * :py:class:`guiqwt.curve.CurveItem`: a curve plot item
    * :py:class:`guiqwt.curve.ErrorBarCurveItem`: a curve plot item with 
      error bars
    * :py:class:`guiqwt.curve.GridItem`
    * :py:class:`guiqwt.curve.ItemListWidget`: base widget implementing the 
      `plot item list panel`
    * :py:class:`guiqwt.curve.PlotItemList`: the `plot item list panel`

``CurveItem`` and ``GridItem`` objects are plot items (derived from 
QwtPlotItem) that may be displayed on a 2D plotting widget like 
:py:class:`guiqwt.curve.CurvePlot` or :py:class:`guiqwt.image.ImagePlot`.

.. seealso::
    
    Module :py:mod:`guiqwt.image`
        Module providing image-related plot items and plotting widgets
        
    Module :py:mod:`guiqwt.plot`
        Module providing ready-to-use curve and image plotting widgets and 
        dialog boxes

Examples
~~~~~~~~

Create a basic curve plotting widget:
    * before creating any widget, a `QApplication` must be instantiated (that 
      is a `Qt` internal requirement):
          
>>> import guidata
>>> app = guidata.qapplication()

    * that is mostly equivalent to the following (the only difference is that 
      the `guidata` helper function also installs the `Qt` translation 
      corresponding to the system locale):
          
>>> from PyQt4.QtGui import QApplication
>>> app = QApplication([])

    * now that a `QApplication` object exists, we may create the plotting 
      widget:
          
>>> from guiqwt.curve import CurvePlot
>>> plot = CurvePlot(title="Example", xlabel="X", ylabel="Y")

Create a curve item:
    * from the associated plot item class (e.g. `ErrorBarCurveItem` to 
      create a curve with error bars): the item properties are then assigned 
      by creating the appropriate style parameters object
      (e.g. :py:class:`guiqwt.styles.ErrorBarParam`)
      
>>> from guiqwt.curve import CurveItem
>>> from guiqwt.styles import CurveParam
>>> param = CurveParam()
>>> param.label = 'My curve'
>>> curve = CurveItem(param)
>>> curve.set_data(x, y)
      
    * or using the `plot item builder` (see :py:func:`guiqwt.builder.make`):
      
>>> from guiqwt.builder import make
>>> curve = make.curve(x, y, title='My curve')

Attach the curve to the plotting widget:
    
>>> plot.add_item(curve)

Display the plotting widget:
    
>>> plot.show()
>>> app.exec_()

Reference
~~~~~~~~~

.. autoclass:: CurvePlot
   :members:
   :inherited-members:
.. autoclass:: CurveItem
   :members:
   :inherited-members:
.. autoclass:: ErrorBarCurveItem
   :members:
   :inherited-members:
.. autoclass:: PlotItemList
   :members:
"""

import sys, numpy as np

from PyQt4.QtGui import (QMenu, QListWidget, QListWidgetItem, QVBoxLayout,
                         QToolBar, QMessageBox, QBrush)
from PyQt4.QtCore import Qt, QPoint, QPointF, QLineF, SIGNAL, QRectF, QLine
from PyQt4.Qwt5 import QwtPlotCurve, QwtPlotGrid, QwtPlotItem, QwtScaleMap

from guidata.utils import assert_interfaces_valid, update_dataset
from guidata.configtools import get_icon, get_image_layout
from guidata.qthelpers import create_action, add_actions

# Local imports
from guiqwt.config import CONF, _
from guiqwt.interfaces import (IBasePlotItem, IDecoratorItemType,
                               ISerializableType, ICurveItemType,
                               ITrackableItemType, IPanel)
from guiqwt.panels import PanelWidget, ID_ITEMLIST
from guiqwt.baseplot import EnhancedQwtPlot
from guiqwt.styles import GridParam, CurveParam, ErrorBarParam, SymbolParam
from guiqwt.shapes import Marker
from guiqwt.signals import (SIG_ACTIVE_ITEM_CHANGED, SIG_ITEMS_CHANGED,
                            SIG_ITEM_REMOVED, SIG_AXIS_DIRECTION_CHANGED)


def seg_dist(P, P0, P1):
    """
    Return distance between point P and segment (P0, P1)
    If P orthogonal projection on (P0, P1) is outside segment bounds, return
    either distance to P0 or to P1 (the closest one)
    P, P0, P1: QPointF instances
    """
    u = QLineF(P0, P).length()
    if P0 == P1:
        return u
    else:
        angle = QLineF(P0, P).angleTo(QLineF(P0, P1))*np.pi/180
        projection = u*np.cos(angle)
        if  projection > QLineF(P0, P1).length():
            return QLineF(P1, P).length()
        elif projection < 0:
            return QLineF(P0, P).length()
        else:
            return abs(u*np.sin(angle))

def test_seg_dist():
    print seg_dist(QPointF(200, 100), QPointF(150, 196), QPointF(250, 180))
    print seg_dist(QPointF(200, 100), QPointF(190, 196), QPointF(210, 180))
    print seg_dist(QPointF(201, 105), QPointF(201, 196), QPointF(201, 180))

def norm2(v):
    return (v**2).sum(axis=1)

def seg_dist_v(P, X0, Y0, X1, Y1):
    """Version vectorielle de seg_dist"""
    V = np.zeros((X0.shape[0], 2), float)
    PP = np.zeros((X0.shape[0], 2), float)
    PP[:, 0] = X0
    PP[:, 1] = Y0    
    V[:, 0] = X1-X0
    V[:, 1] = Y1-Y0
    dP = np.array(P).reshape(1, 2) - PP
    nV = np.sqrt(norm2(V))
    w2 = V/nV[:, np.newaxis]
    w = np.array([ -w2[:,1], w2[:,0] ]).T
    distances = np.fabs((dP*w).sum(axis=1))
    ix = distances.argmin()
    return ix, distances[ix]

def test_seg_dist_v():
    """Test de seg_dist_v"""
    a=(np.arange(10.)**2).reshape(5, 2)
    ix, dist = seg_dist_v((2.1, 3.3), a[:-1, 0], a[:-1, 1],
                          a[1:, 0], a[1:, 1])
    print ix, dist
    assert ix == 0

if __name__ == "__main__":
    test_seg_dist_v()
    test_seg_dist()


SELECTED_SYMBOL_PARAM = SymbolParam()
SELECTED_SYMBOL_PARAM.read_config(CONF, "plot", "selected_curve_symbol")
SELECTED_SYMBOL = SELECTED_SYMBOL_PARAM.build_symbol()


class GridItem(QwtPlotGrid):
    """
    Construct a grid `plot item` with the parameters *gridparam*
    (see :py:class:`guiqwt.styles.GridParam`)
    """
    __implements__ = (IBasePlotItem,)
    
    _readonly = True
    
    def __init__(self, gridparam=None):
        super(GridItem, self).__init__()
        if gridparam is None:
            self.gridparam = GridParam(title=_("Grid"), icon="grid.png")
        else:
            self.gridparam = gridparam
        self.selected = False
        self.immutable = True # set to false to allow moving points around
        self.update_params() # won't work completely because it's not yet
        # attached to plot (actually, only canvas background won't be updated)

    def types(self):
        return (IDecoratorItemType,)
    
    def attach(self, plot):
        """Reimplemented to update plot canvas background"""
        super(GridItem, self).attach(plot)
        self.update_params()

    def set_readonly(self, state):
        """Set object read-only state"""
        self._readonly = state
        
    def is_readonly(self):
        """Return object read-only state"""
        return self._readonly

    def can_select(self):
        return False
    def can_resize(self):
        return False
    def can_rotate(self):
        return False
    def can_move(self):
        return False

    def select(self):
        """Select item"""
        pass
    
    def unselect(self):
        """Unselect item"""
        pass

    def hit_test(self, pos):
        return sys.maxint, 0, False, None

    def move_local_point_to(self, handle, pos ):
        pass

    def move_local_shape(self, old_pos, new_pos):
        pass
        
    def move_with_selection(self, dx, dy):
        pass

    def update_params(self):
        self.gridparam.update_grid(self)

    def get_item_parameters(self, itemparams):
        itemparams.add("GridParam", self, self.gridparam)
    
    def set_item_parameters(self, itemparams):
        self.gridparam = itemparams.get("GridParam")
        self.gridparam.update_grid(self)

assert_interfaces_valid(GridItem)


class CurveItem(QwtPlotCurve):
    """
    Construct a curve `plot item` with the parameters *curveparam*
    (see :py:class:`guiqwt.styles.CurveParam`)
    """
    __implements__ = (IBasePlotItem,)
    
    _readonly = False
    
    def __init__(self, curveparam=None):
        super(CurveItem, self).__init__()
        if curveparam is None:
            self.curveparam = CurveParam(_("Curve"), icon='curve.png')
        else:
            self.curveparam = curveparam
        self.selected = False
        self.immutable = True # set to false to allow moving points around
        self._x = None
        self._y = None
        self.update_params()
        
    def types(self):
        return (ICurveItemType, ITrackableItemType, ISerializableType)

    def can_select(self):
        return True
    def can_resize(self):
        return False
    def can_rotate(self):
        return False
    def can_move(self):
        return False

    def __reduce__(self):
        state = (self.curveparam, self._x, self._y, self.z())
        res = ( CurveItem, (), state )
        return res

    def __setstate__(self, state):
        param, x, y, z = state
        self.curveparam = param
        self.set_data(x, y)
        self.setZ(z)
        self.update_params()

    def set_readonly(self, state):
        """Set object readonly state"""
        self._readonly = state
        
    def is_readonly(self):
        """Return object readonly state"""
        return self._readonly

    def invalidate_plot(self):
        plot = self.plot()
        if plot is not None:
            plot.invalidate()

    def select(self):
        """Select item"""
        self.selected = True
        self.setSymbol(SELECTED_SYMBOL)
        self.invalidate_plot()
    
    def unselect(self):
        """Unselect item"""
        self.selected = False
        # Restoring initial curve parameters:
        self.curveparam.update_curve(self)
        self.invalidate_plot()

    def get_data(self):
        """Return curve data x, y (NumPy arrays)"""
        return self._x, self._y

    def set_data(self, x, y):
        """
        Set curve data:
            * x: NumPy array
            * y: NumPy array
        """
        self._x = np.array(x, copy=False)
        self._y = np.array(y, copy=False)
        self.setData(self._x, self._y)
        
    def is_empty(self):
        """Return True if item data is empty"""
        return self._x is None or self._y is None or self._y.size == 0

    def hit_test(self, pos):
        """Calcul de la distance d'un point à une courbe
        renvoie (dist, handle, inside)"""
        if self.is_empty():
            return sys.maxint, 0, False, None
        plot = self.plot()
        ax = self.xAxis()
        ay = self.yAxis()
        px = plot.invTransform(ax, pos.x())
        py = plot.invTransform(ay, pos.y())
        # On cherche les 4 points qui sont les plus proches en X et en Y
        # avant et après ie tels que p1x < x < p2x et p3y < y < p4y
        dx = 1/(self._x-px)
        dy = 1/(self._y-py)
        i0 = dx.argmin()
        i1 = dx.argmax()
        i2 = dy.argmin()
        i3 = dy.argmax()
        t = np.array((i0, i1, i2, i3))
        t2 = (t+1).clip(0, self._x.shape[0]-1)
        i, _d = seg_dist_v((px, py), self._x[t], self._y[t],
                           self._x[t2], self._y[t2])
        i = t[i]
        # Recalcule la distance dans le répère du widget
        p0x = plot.transform(ax, self._x[i])
        p0y = plot.transform(ay, self._y[i])
        if i+1 >= self._x.shape[0]:
            p1x = p0x
            p1y = p0y
        else:
            p1x = plot.transform(ax, self._x[i+1])
            p1y = plot.transform(ay, self._y[i+1])
        distance = seg_dist(QPointF(pos), QPointF(p0x, p0y), QPointF(p1x, p1y))
        return distance, i, False, None
    
    def get_closest_coordinates(self, x, y):
        """Renvoie les coordonnées (x',y') du point le plus proche de (x,y)
        Méthode surchargée pour ErrorBarSignalCurve pour renvoyer
        les coordonnées des pointes des barres d'erreur"""
        plot = self.plot()
        ax = self.xAxis()
        ay = self.yAxis()
        xc = plot.transform(ax, x)
        yc = plot.transform(ay, y)
        _distance, i, _inside, _other = self.hit_test(QPoint(xc, yc))
        x = self.x(i)
        y = self.y(i)
        return x, y

    def get_coordinates_label(self, xc, yc):
        title = self.title().text()
        return "%s:<br>x = %f<br>y = %f" % (title, xc, yc)

    def get_closest_x(self, xc):
        # We assume X is sorted, otherwise we'd need :
        # argmin(abs(x-xc))
        i = self._x.searchsorted(xc)
        if i>0:
            if np.fabs(self._x[i-1]-xc) < np.fabs(self._x[i]-xc):
                return self._x[i-1], self._y[i-1]
        return self._x[i], self._y[i]

    def canvas_to_axes(self, pos):
        plot = self.plot()
        ax = self.xAxis()
        ay = self.yAxis()
        return plot.invTransform(ax, pos.x()), plot.invTransform(ay, pos.y())

    def move_local_point_to(self, handle, pos):
        if self.immutable:
            return
        if handle < 0 or handle > self._x.shape[0]:
            return
        x, y = self.canvas_to_axes(pos)
        self._x[handle] = x
        self._y[handle] = y
        self.setData(self._x, self._y)
        self.plot().replot()

    def move_local_shape(self, old_pos, new_pos):
        """Translate the shape such that old_pos becomes new_pos
        in canvas coordinates"""
        nx, ny = self.canvas_to_axes(new_pos)
        ox, oy = self.canvas_to_axes(old_pos)
        self._x += (nx-ox)
        self._y += (ny-oy)
        self.setData(self._x, self._y)
        
    def move_with_selection(self, dx, dy):
        """
        Translate the shape together with other selected items
        dx, dy: translation in plot coordinates
        """
        self._x += dx
        self._y += dy
        self.setData(self._x, self._y)

    def update_params(self):
        self.curveparam.update_curve(self)
        if self.selected:
            self.select()

    def get_item_parameters(self, itemparams):
        itemparams.add("CurveParam", self, self.curveparam)
    
    def set_item_parameters(self, itemparams):
        update_dataset(self.curveparam, itemparams.get("CurveParam"),
                       visible_only=True)
        self.update_params()

assert_interfaces_valid(CurveItem)


def _transform(map,v):
    return QwtScaleMap.transform(map,v)
vmap = np.vectorize(_transform)

class ErrorBarCurveItem(CurveItem):
    """
    Construct an error-bar curve `plot item` 
    with the parameters *errorbarparam*
    (see :py:class:`guiqwt.styles.ErrorBarParam`)
    """
    def __init__(self, curveparam=None, errorbarparam=None):
        if errorbarparam is None:
            self.errorbarparam = ErrorBarParam(_("Error bars"),
                                               icon='errorbar.png')
        else:
            self.errorbarparam = errorbarparam
        super(ErrorBarCurveItem, self).__init__(curveparam)
        self._dx = None
        self._dy = None
        
    def unselect(self):
        """Unselect item"""
        super(ErrorBarCurveItem, self).unselect()
        self.errorbarparam.update_curve(self)

    def get_data(self):
        """
        Return error-bar curve data: x, y, dx, dy
            * x: NumPy array
            * y: NumPy array
            * dx: float or NumPy array (non-constant error bars)
            * dy: float or NumPy array (non-constant error bars)
        """
        return self._x, self._y, self._dx, self._dy

    def set_data(self, x, y, dx, dy):
        """
        Set error-bar curve data:
            * x: NumPy array
            * y: NumPy array
            * dx: float or NumPy array (non-constant error bars)
            * dy: float or NumPy array (non-constant error bars)
        """
        CurveItem.set_data(self, x, y)
        if dx is not None:
            dx = np.array(dx, copy=False)
            if dx.size == 0:
                dx = None
        if dy is not None:
            dy = np.array(dy, copy=False)
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


#===============================================================================
# Plot Widget
#===============================================================================
class ItemListWidget(QListWidget):
    """
    PlotItemList
    List of items attached to plot
    """
    def __init__(self, parent):
        super(ItemListWidget, self).__init__(parent)
        
        self.manager = None
        self.plot = None # the default plot...
        self.items = []
        
        self.connect(self, SIGNAL('currentRowChanged(int)'),
                     self.current_row_changed)
        self.connect(self, SIGNAL('itemChanged(QListWidgetItem*)'),
                     self.item_changed)
        self.connect(self, SIGNAL('itemSelectionChanged()'),
                     self.refresh_actions)
        self.connect(self, SIGNAL('itemSelectionChanged()'),
                     self.selection_changed)
        
        self.setWordWrap(True)
        self.setMinimumWidth(140)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        
        # Setup context menu
        self.menu = QMenu(self)
        self.menu_actions = self.setup_actions()
        self.refresh_actions()
        add_actions(self.menu, self.menu_actions)

    def register_panel(self, manager):
        self.manager = manager

        for plot in self.manager.get_plots():
            self.connect(plot, SIG_ITEMS_CHANGED, self.items_changed)
            self.connect(plot, SIG_ACTIVE_ITEM_CHANGED, self.items_changed)
        self.plot = self.manager.get_plot()

    def contextMenuEvent(self, event):
        """Override Qt method"""
        self.refresh_actions()
        self.menu.popup(event.globalPos())
                     
    def setup_actions(self):
        self.movedown_ac = create_action(self, _("Move to back"),
                                     icon=get_icon('arrow_down.png'),
                                     triggered=lambda: self.move_item("down"))
        self.moveup_ac = create_action(self, _("Move to front"),
                                       icon=get_icon('arrow_up.png'),
                                       triggered=lambda: self.move_item("up"))
        settings_ac = create_action(self, _("Parameters..."),
                    icon=get_icon('settings.png'),
                    triggered=self.edit_plot_parameters )
        self.remove_ac = create_action(self, _("Remove"),
                                       icon=get_icon('trash.png'),
                                       triggered=self.remove_item)
        return [self.moveup_ac, self.movedown_ac, None,
                settings_ac, self.remove_ac]

    def edit_plot_parameters(self):
        self.plot.edit_plot_parameters("item")
    
    def __is_selection_contiguous(self):
        indexes = sorted([self.row(lw_item) for lw_item
                          in self.selectedItems()])
        return len(indexes) <= 1 or range(indexes[0], indexes[-1]+1) == indexes
        
    def get_selected_items(self):
        """Return selected QwtPlot items
        Warning: this is not the same as self.plot.get_selected_items
        --> some items could appear in itemlist without being registered in 
            plot widget items (in particular, some items could be selected in 
            itemlist without being selected in plot widget)
        """
        return [self.items[self.row(lw_item)]
                for lw_item in self.selectedItems()]
        
    def refresh_actions(self):
        is_selection = len(self.selectedItems()) > 0
        for action in self.menu_actions:
            if action is not None:
                action.setEnabled(is_selection)
        if is_selection:
            remove_state = True
            for item in self.get_selected_items():
                remove_state = remove_state and not item.is_readonly()
            self.remove_ac.setEnabled(remove_state)
            for action in [self.moveup_ac, self.movedown_ac]:
                action.setEnabled(self.__is_selection_contiguous())            
        
    def __get_item_icon(self, item):
        from guiqwt.label import LegendBoxItem, LabelItem
        from guiqwt.annotations import (AnnotatedShape, AnnotatedRectangle,
                                        AnnotatedCircle, AnnotatedEllipse,
                                        AnnotatedPoint, AnnotatedSegment)
        from guiqwt.shapes import (SegmentShape, RectangleShape, EllipseShape,
                                   PointShape, PolygonShape, Axes,
                                   XRangeSelection)
        from guiqwt.image import (BaseImageItem, Histogram2DItem,
                                  ImageFilterItem)
        from guiqwt.histogram import HistogramItem

        icon_name = 'item.png'
        for klass, icon in ((HistogramItem, 'histogram.png'),
                            (ErrorBarCurveItem, 'errorbar.png'),
                            (CurveItem, 'curve.png'),
                            (GridItem, 'grid.png'),
                            (LegendBoxItem, 'legend.png'),
                            (LabelItem, 'label.png'),
                            (AnnotatedSegment, 'segment.png'),
                            (AnnotatedPoint, 'point_shape.png'),
                            (AnnotatedCircle, 'circle.png'),
                            (AnnotatedEllipse, 'ellipse_shape.png'),
                            (AnnotatedRectangle, 'rectangle.png'),
                            (AnnotatedShape, 'annotation.png'),
                            (SegmentShape, 'segment.png'),
                            (RectangleShape, 'rectangle.png'),
                            (PointShape, 'point_shape.png'),
                            (EllipseShape, 'ellipse_shape.png'),
                            (Axes, 'gtaxes.png'),
                            (Marker, 'marker.png'),
                            (XRangeSelection, 'xrange.png'),
                            (PolygonShape, 'freeform.png'),
                            (Histogram2DItem, 'histogram2d.png'),
                            (ImageFilterItem, 'funct.png'),
                            (BaseImageItem, 'image.png'),):
            if isinstance(item, klass):
                icon_name = icon
                break
        return get_icon(icon_name)
        
    def items_changed(self, plot):
        """Plot items have changed"""
        active_plot = self.manager.get_active_plot()
        if active_plot is not plot:
            return
        self.plot = plot
        _block = self.blockSignals(True)
        active = plot.get_active_item()
        self.items = plot.get_items(z_sorted=True)
        self.clear()
        for item in self.items:
            title = item.title().text()
            lw_item = QListWidgetItem(self.__get_item_icon(item), title, self)
            lw_item.setCheckState(Qt.Checked if item.isVisible()
                                  else Qt.Unchecked)
            lw_item.setSelected(item.selected)
            font = lw_item.font()
            if item is active:
                font.setItalic(True)
            else:
                font.setItalic(False)
            lw_item.setFont(font)
            self.addItem(lw_item)
        self.refresh_actions()
        self.blockSignals(_block)
            
    def current_row_changed(self, index):
        """QListWidget current row has changed"""
        if index == -1:
            return
        item = self.items[index]
        if not item.can_select():
            item = None
        if item is None:
            self.plot.replot()
                
    def selection_changed(self):
        items = self.get_selected_items()
        self.plot.select_some_items(items)
        self.plot.replot()
        
    def item_changed(self, listwidgetitem):
        """QListWidget item has changed"""
        item = self.items[self.row(listwidgetitem)]
        visible = listwidgetitem.checkState() == Qt.Checked
        if visible != item.isVisible():
            item.setVisible(visible)
            self.plot.replot()
    
    def move_item(self, direction):
        """Move item to the background/foreground
        Works only for contiguous selection
        -> 'refresh_actions' method should guarantee that"""
        items = self.get_selected_items()
        if direction == 'up':
            self.plot.move_up(items)
        else:
            self.plot.move_down(items)
        # Re-select items which can't be selected in plot widget but can be 
        # selected in ItemListWidget:
        for item in items:
            lw_item = self.item(self.items.index(item))
            if not lw_item.isSelected():
                lw_item.setSelected(True)
        self.plot.replot()
        
    def remove_item(self):
        if len(self.selectedItems()) == 1:
            message = _("Do you really want to remove this item?")
        else:
            message = _("Do you really want to remove selected items?")
        answer = QMessageBox.warning(self, _("Remove"), message,
                                     QMessageBox.Yes | QMessageBox.No)
        if answer == QMessageBox.Yes:
            items = self.get_selected_items()
            self.plot.del_items(items)
            self.plot.replot()
            for item in items:
                self.parent().emit(SIG_ITEM_REMOVED, item)
        

class PlotItemList(PanelWidget):
    """Construct the `plot item list panel`"""
    __implements__ = (IPanel,)
    PANEL_ID = ID_ITEMLIST
    
    def __init__(self, parent):
        super(PlotItemList, self).__init__(parent)
        widget_title = _("Item list")
        widget_icon = "item_list.png"
        self.manager = None
        
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        
        style = "<span style=\'color: #444444\'><b>%s</b></span>"
        layout, _label = get_image_layout(widget_icon, style % widget_title,
                                          alignment=Qt.AlignCenter)
        vlayout.addLayout(layout)
        self.listwidget = ItemListWidget(self)
        vlayout.addWidget(self.listwidget)
        
        toolbar = QToolBar(self)
        vlayout.addWidget(toolbar)
        add_actions(toolbar, self.listwidget.menu_actions)
        
        self.setWindowIcon(get_icon(widget_icon))
        self.setWindowTitle(widget_title)

    def register_panel(self, manager):
        """Register panel to plot manager"""
        self.manager = manager
        self.listwidget.register_panel(manager)

assert_interfaces_valid(PlotItemList)


class CurvePlot(EnhancedQwtPlot):
    """
    Construct a 2D curve plotting widget 
    (this class inherits :py:class:`guiqwt.baseplot.EnhancedQwtPlot`)
        * parent: parent widget
        * title: plot title
        * xlabel: (bottom axis title, top axis title) or bottom axis title only
        * ylabel: (left axis title, right axis title) or left axis title only
        * gridparam: GridParam instance
    """
    AUTOSCALE_TYPES = (CurveItem,)
    def __init__(self, parent=None, title=None, xlabel=None, ylabel=None,
                 gridparam=None, section="plot"):
        super(CurvePlot, self).__init__(parent, section)

        self.axes_reverse = [False]*4
        
        self.set_titles(title=title, xlabel=xlabel, ylabel=ylabel)
                
        self.antialiased = False

        self.set_antialiasing(CONF.get(section, "antialiasing"))
        
        # Installing our own event filter:
        # (PyQwt's event filter does not fit our needs)
        self.canvas().installEventFilter(self.filter)
        self.canvas().setMouseTracking(True)
    
        self.cross_marker = Marker()
        self.curve_marker = Marker(label_cb=self.get_coordinates_str,
                                   constraint_cb=self.on_active_curve)
        self.cross_marker.set_style(section, "marker/cross")
        self.curve_marker.set_style(section, "marker/curve")
        self.cross_marker.setVisible(False)
        self.curve_marker.setVisible(False)
        self.cross_marker.attach(self)
        self.curve_marker.attach(self)

        # Background color
        self.setCanvasBackground(Qt.white)        
        
        self.curve_pointer = False
        self.canvas_pointer = False
        
        # Setting up grid
        if gridparam is None:
            gridparam = GridParam(title=_("Grid"), icon="grid.png")
            gridparam.read_config(CONF, section, "grid")
        self.grid = GridItem(gridparam)
        self.add_item(self.grid, z=-1)

    #---- Private API ----------------------------------------------------------
    def __del__(self):
        # Sometimes, an obscure exception happens when we quit an application
        # because if we don't remove the eventFilter it can still be called
        # after the filter object has been destroyed by Python.
        canvas = self.canvas()
        if canvas:
            canvas.removeEventFilter(self.filter)

    # generic helper methods        
    def canvas2plotitem(self, plot_item, x_canvas, y_canvas):
        return (self.invTransform(plot_item.xAxis(), x_canvas),
                self.invTransform(plot_item.yAxis(), y_canvas))
    def plotitem2canvas(self, plot_item, x, y):
        return (self.transform(plot_item.xAxis(), x),
                self.transform(plot_item.yAxis(), y))

    def on_active_curve(self, marker, x, y):
        curve = self.get_last_active_item(ITrackableItemType)
        if curve:
            x, y = curve.get_closest_coordinates(x, y)
        return x, y
    
    def get_coordinates_str(self, marker, x, y):
        title = _("Grid")
        item = self.get_last_active_item(ITrackableItemType)
        if item:
            return item.get_coordinates_label(x, y)
        return "<b>%s</b><br>x = %f<br>y = %f" % (title, x, y)

    def set_marker_axes(self):
        curve = self.get_last_active_item(ITrackableItemType)
        if curve:
            self.cross_marker.setAxis(curve.xAxis(), curve.yAxis())
            self.curve_marker.setAxis(curve.xAxis(), curve.yAxis())
    
    def do_move_marker(self, event):
        pos = event.pos()
        self.set_marker_axes()
        if event.modifiers() & Qt.ShiftModifier or self.curve_pointer :
            self.curve_marker.setZ(self.get_max_z()+1)
            self.cross_marker.setVisible(False)
            self.curve_marker.setVisible(True)
            self.curve_marker.move_local_point_to(0, pos)
            self.replot()
            #self.move_curve_marker(self.curve_marker, xc, yc)
        elif event.modifiers() & Qt.AltModifier or self.canvas_pointer:
            self.cross_marker.setZ(self.get_max_z()+1)
            self.cross_marker.setVisible(True)
            self.curve_marker.setVisible(False)
            self.cross_marker.move_local_point_to(0, pos)
            self.replot()
            #self.move_canvas_marker(self.cross_marker, xc, yc)
        else:
            vis_cross = self.cross_marker.isVisible()
            vis_curve = self.curve_marker.isVisible()
            self.cross_marker.setVisible(False)
            self.curve_marker.setVisible(False)
            if vis_cross or vis_curve:
                self.replot()
        
    def do_pan_view(self, dx, dy):
        """
        Translate the active axes by dx, dy
        dx, dy are tuples composed of (initial pos, dest pos)
        """
        auto = self.autoReplot()
        self.setAutoReplot(False)
        xaxis, yaxis = self.get_active_axes()
        active_axes = [ (dx, xaxis),
                        (dy, yaxis) ]
        for (x1, x0, _, w), k in active_axes:
            axis = self.axisScaleDiv(k)
            # pour les axes logs on bouge lbound et hbound relativement
            # à l'inverse du delta aux bords de l'axe
            # pour des axes lineaires pos0 et pos1 doivent être égaux
            pos0 = self.invTransform(k, x1-x0)-self.invTransform(k, 0)
            pos1 = self.invTransform(k, x1-x0+w)-self.invTransform(k, w)
            lbound = axis.lowerBound()
            hbound = axis.upperBound()
            self.setAxisScale(k, lbound-pos0, hbound-pos1)
        self.setAutoReplot(auto)
        self.replot()

    def do_zoom_view(self, dx, dy, lock_aspect_ratio=False):
        """
        Change the scale of the active axes (zoom/dezoom) according to dx, dy
        dx, dy are tuples composed of (initial pos, dest pos)
        We try to keep initial pos fixed on the canvas as the scale changes
        """
        auto = self.autoReplot()
        self.setAutoReplot(False)
        dx = (-1,) + dx
        dy = (1,) + dy
        if lock_aspect_ratio:
            sens, x1, x0, s, w = dx
            F = 1+3*sens*float(x1-x0)/w
        xaxis, yaxis = self.get_active_axes()
        active_axes = [ (dx, xaxis),
                        (dy, yaxis) ]
        for (sens, x1, x0, s, w), k in active_axes:
            axis = self.axisScaleDiv(k)
            lbound = axis.lowerBound()
            hbound = axis.upperBound()
            orig = self.invTransform(k, s)
            rng = float(hbound-lbound)
            if not lock_aspect_ratio:
                F = 1+3*sens*float(x1-x0)/w
            l_new = orig-F*(orig-lbound)
            if F*rng == 0:
                continue
            self.setAxisScale(k, l_new, l_new + F*rng)
        self.setAutoReplot(auto)
        self.replot()
        
    def do_zoom_rect_view(self, start, end):
        x1, y1 = start.x(), start.y()
        x2, y2 = end.x(), end.y()
        xaxis, yaxis = self.get_active_axes()
        active_axes = [ (x1, x2, xaxis),
                        (y1, y2, yaxis) ]
        for h1, h2, k in active_axes:
            o1 = self.invTransform(k, h1)
            o2 = self.invTransform(k, h2)
            if o1 > o2:
                o1, o2 = o2, o1
            if o1 == o2:
                continue
            if self.get_axis_direction(k):
                o1, o2 = o2, o1
            self.setAxisScale(k, o1, o2)
        self.replot()

    #---- EnhancedQwtPlot API --------------------------------------------------
    def get_axis_title(self, axis):
        """
        Reimplement EnhancedQwtPlot method
        
        Return axis title
            * axis: 'bottom', 'left', 'top' or 'right'
        """
        if axis in self.AXES:
            axis = self.AXES[axis]
        return super(CurvePlot, self).get_axis_title(axis)
        
    def set_axis_title(self, axis, title):
        """
        Reimplement EnhancedQwtPlot method
        
        Set axis title
            * axis: 'bottom', 'left', 'top' or 'right'
            * title: string
        """
        if axis in self.AXES:
            axis = self.AXES[axis]
        super(CurvePlot, self).set_axis_title(axis, title)

    def set_axis_font(self, axis, font):
        """
        Reimplement EnhancedQwtPlot method
        
        Set axis font
            * axis: 'bottom', 'left', 'top' or 'right'
            * font: QFont instance
        """
        if axis in self.AXES:
            axis = self.AXES[axis]
        super(CurvePlot, self).set_axis_font(axis, font)
    
    def set_axis_color(self, axis, color):
        """
        Reimplement EnhancedQwtPlot method
        
        Set axis color
            * axis: 'bottom', 'left', 'top' or 'right'
            * color: color name (string) or QColor instance
        """
        if axis in self.AXES:
            axis = self.AXES[axis]
        super(CurvePlot, self).set_axis_color(axis, color)

    def add_item(self, item, z=None):
        """
        Add a *plot item* instance to this *plot widget*
            * item: QwtPlotItem (PyQt4.Qwt5) object implementing
              the IBasePlotItem interface (guiqwt.interfaces)
            * z: item's z order (None -> z = max(self.get_items())+1)
        """
        if isinstance(item, QwtPlotCurve):
            item.setRenderHint(QwtPlotItem.RenderAntialiased, self.antialiased)
        super(CurvePlot,self).add_item(item, z)

    def del_all_items(self, except_grid=True):
        """Del all items, eventually (default) except grid"""
        items = [item for item in self.items
                 if not except_grid or item is not self.grid]
        self.del_items(items)
    
    def set_active_item(self, item):
        """Override base set_active_item to change the grid's
        axes according to the selected item"""
        old_active = self.active_item
        super(CurvePlot, self).set_active_item(item)
        if item is not None and old_active is not item:
            self.grid.setAxis(item.xAxis(), item.yAxis())

    def get_plot_parameters(self, key, itemparams):
        if key == "grid":
            self.grid.gridparam.update_param(self.grid)
            itemparams.add("GridParam", self, self.grid.gridparam)
        else:
            super(CurvePlot, self).get_plot_parameters(key, itemparams)

    def set_item_parameters(self, itemparams):
        # Grid style
        dataset = itemparams.get("GridParam")
        if dataset is not None:
            dataset.update_grid(self.grid)
            self.grid.gridparam = dataset
        super(CurvePlot, self).set_item_parameters(itemparams)
    
    def do_autoscale(self, replot=True):
        """Do autoscale on all axes"""
        rect = None
        for item in self.get_items():
            if isinstance(item, self.AUTOSCALE_TYPES) and not item.is_empty() \
               and item.isVisible():
                bounds = item.boundingRect()
                if rect is None:
                    rect = bounds
                else:
                    rect = rect.united(bounds)
        if rect is not None:
            x0, x1 = rect.left(), rect.right()
            y0, y1 = rect.top(), rect.bottom()
            if x0 == x1: # same behavior as MATLAB
                x0 -= 1
                x1 += 1
            if y0 == y1: # same behavior as MATLAB
                y0 -= 1
                y1 += 1
            self.set_plot_limits(x0, x1, y0, y1)
            if replot:
                self.replot()
            
    #---- Public API -----------------------------------------------------------    
    def get_axis_direction(self, axis):
        """
        Return axis direction of increasing values
            * axis: axis id (QwtPlot.yLeft, QwtPlot.xBottom, ...)
              or string: 'bottom', 'left', 'top' or 'right'
        """
        axis_id = self.AXES.get(axis, axis)
        return self.axes_reverse[axis_id]
            
    def set_axis_direction(self, axis, reverse=False):
        """
        Set axis direction of increasing values
            * axis: axis id (QwtPlot.yLeft, QwtPlot.xBottom, ...)
              or string: 'bottom', 'left', 'top' or 'right'
            * reverse: False (default)
                - x-axis values increase from left to right
                - y-axis values increase from bottom to top
            * reverse: True
                - x-axis values increase from right to left
                - y-axis values increase from top to bottom
        """
        axis_id = self.AXES.get(axis, axis)
        if reverse != self.axes_reverse[axis_id]:
            self.replot()
            self.axes_reverse[axis_id] = reverse
            axis_map = self.canvasMap(axis_id)
            self.setAxisScale(axis_id, axis_map.s2(), axis_map.s1())
            self.updateAxes()
            self.emit(SIG_AXIS_DIRECTION_CHANGED, self, axis_id)
            
    def set_titles(self, title=None, xlabel=None, ylabel=None):
        """
        Set plot and axes titles at once
            * title: plot title
            * xlabel: (bottom axis title, top axis title) 
              or bottom axis title only
            * ylabel: (left axis title, right axis title) 
              or left axis title only
        """
        if title is not None:
            self.set_title(title)
        if xlabel is not None:
            if isinstance(xlabel, basestring):
                xlabel = (xlabel, "")
            for label, axis in zip(xlabel, ("bottom", "top")):
                if label:
                    self.set_axis_title(axis, label)
        if ylabel is not None:
            if isinstance(ylabel, basestring):
                ylabel = (ylabel, "")
            for label, axis in zip(ylabel, ("left", "right")):
                if label:
                    self.set_axis_title(axis, label)
    
    def set_pointer(self, pointer_type):
        """
        Set pointer.
        Valid values of `pointer_type`:
            * None: disable pointer
            * "canvas": enable canvas pointer
            * "curve": enable on-curve pointer
        """
        self.canvas_pointer = False
        self.curve_pointer = False
        if pointer_type == "canvas":
            self.canvas_pointer = True
        elif pointer_type == "curve":
            self.curve_pointer = True

    def set_antialiasing(self, checked):
        """Toggle curve antialiasing"""
        self.antialiased = checked
        for curve in self.itemList():
            if isinstance(curve, QwtPlotCurve):
                curve.setRenderHint(QwtPlotItem.RenderAntialiased,
                                    self.antialiased)

    def set_plot_limits(self, x0, x1, y0, y1):
        """Set plot scale limits"""
        dy = y1-y0
        if self.get_axis_direction(self.yLeft):
            self.setAxisScale(self.yLeft, y0+dy, y0)
        else:
            self.setAxisScale(self.yLeft, y0, y0+dy)
        dx = x1-x0
        if self.get_axis_direction(self.xBottom):
            self.setAxisScale(self.xBottom, x0+dx, x0)
        else:
            self.setAxisScale(self.xBottom, x0, x0+dx)
        self.updateAxes()
        self.emit(SIG_AXIS_DIRECTION_CHANGED, self, self.yLeft)
        self.emit(SIG_AXIS_DIRECTION_CHANGED, self, self.xBottom)
        