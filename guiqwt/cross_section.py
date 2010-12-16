# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.cross_section
--------------------

The `cross_section` module provides cross section related objects:
    * :py:class:`guiqwt.cross_section.XCrossSection`: the X-axis 
      `cross-section panel`
    * :py:class:`guiqwt.cross_section.YCrossSection`: the Y-axis 
      `cross-section panel`
    * and other related objects which are exclusively used by the cross-section 
      panels

Example
~~~~~~~

Simple cross-section demo:

.. literalinclude:: ../guiqwt/tests/cross_section.py

Reference
~~~~~~~~~

.. autoclass:: XCrossSection
   :members:
   :inherited-members:
.. autoclass:: YCrossSection
   :members:
   :inherited-members:
"""

import weakref

from PyQt4.QtGui import (QVBoxLayout, QSizePolicy, QHBoxLayout, QToolBar,
                         QSpacerItem, QFileDialog, QMessageBox)
from PyQt4.QtCore import QSize, QPoint, Qt, SIGNAL

import numpy as np

from guidata.utils import assert_interfaces_valid
from guidata.configtools import get_icon
from guidata.qthelpers import create_action, add_actions, get_std_icon

# Local imports
from guiqwt.config import CONF, _
from guiqwt.interfaces import (ICSImageItemType, IPanel, IBasePlotItem,
                               ICurveItemType)
from guiqwt.panels import PanelWidget, ID_XCS, ID_YCS
from guiqwt.curve import CurvePlot, CurveItem
from guiqwt.image import ImagePlot
from guiqwt.styles import CurveParam
from guiqwt.tools import SelectTool, BasePlotMenuTool, AntiAliasingTool
from guiqwt.signals import (SIG_MARKER_CHANGED, SIG_PLOT_LABELS_CHANGED,
                            SIG_ANNOTATION_CHANGED, SIG_AXIS_DIRECTION_CHANGED,
                            SIG_ITEMS_CHANGED, SIG_ACTIVE_ITEM_CHANGED,
                            SIG_LUT_CHANGED)
from guiqwt.plot import PlotManager
from guiqwt.builder import make


class CrossSectionItem(CurveItem):
    """A Qwt item representing cross section data"""
    __implements__ = (IBasePlotItem,)
    _inverted = None
    
    def __init__(self, curveparam=None):
        super(CrossSectionItem, self).__init__(curveparam)
        self.perimage_mode = True
        self.autoscale_mode = True
        self.apply_lut = False
        self.source = None

    def set_source_image(self, src):
        """
        Set source image
        (source: object with methods 'get_xsection' and 'get_ysection',
         e.g. objects derived from guiqwt.image.BaseImageItem)
        """
        self.source = weakref.ref(src)
        
    def get_source_image(self):
        if self.source is not None:
            return self.source()

    def get_cross_section(self, obj):
        """Get cross section data from source image"""
        raise NotImplementedError
        
    def update_item(self, obj):
        plot = self.plot()
        if not plot:
            return
        source = self.get_source_image()
        if source is None or not plot.isVisible():
            return
        sectx, secty = self.get_cross_section(obj)
        if secty.size == 0 or np.all(np.isnan(secty)):
            sectx, secty = np.array([]), np.array([])
        if self._inverted:
            self.set_data(secty, sectx)
        else:
            self.set_data(sectx, secty)
        if not self.autoscale_mode:
            self.update_scale()

    def update_scale(self):
        raise NotImplementedError

assert_interfaces_valid(CrossSectionItem)


def get_image_data(plot, p0, p1, apply_lut=False):
    """
    Save rectangular plot area
    p0, p1: resp. top left and bottom right points (QPoint objects)
    """
    from guiqwt.image import (ImageItem, XYImageItem, TrImageItem,
                              get_image_from_plot, get_plot_source_rect)
                           
    items = [item for item in plot.items if isinstance(item, ImageItem)
             and not isinstance(item, XYImageItem)]
    if not items:
        return
    _src_x, _src_y, src_w, src_h = get_plot_source_rect(plot, p0, p1)
    trparams = [item.get_transform() for item in items
                if isinstance(item, TrImageItem)]
    if trparams:
        src_w /= max([dx for _x, _y, _angle, dx, _dy, _hf, _vf in trparams])
        src_h /= max([dy for _x, _y, _angle, _dx, dy, _hf, _vf in trparams])
    return get_image_from_plot(plot, p0, p1, src_w, src_h, apply_lut=apply_lut)


def get_rectangular_area(obj):
    """
    Return rectangular area covered by object
    
    Return None if object does not support this feature 
    (like markers, points, ...)
    """
    try:
        return obj.get_rect()
    except AttributeError:
        return

def get_object_coordinates(obj):
    """Return Marker or PointShape/AnnotatedPoint object coordinates"""
    try:
        return obj.get_pos()
    except AttributeError:
        return obj.xValue(), obj.yValue()

def get_plot_x_section(obj, apply_lut=False):
    """
    Return plot cross section along x-axis,
    at the y value defined by 'obj', a Marker/AnnotatedPoint object
    """
    _x0, y0 = get_object_coordinates(obj)
    plot = obj.plot()
    xmap = plot.canvasMap(plot.AXES["bottom"])
    xc0, xc1 = xmap.p1(), xmap.p2()
    _xc0, yc0 = obj.axes_to_canvas(0, y0)
    if plot.get_axis_direction("left"):
        yc1 = yc0+1
    else:
        yc1 = yc0-1
    try:
        data = get_image_data(plot, QPoint(xc0, yc0), QPoint(xc1, yc1),
                              apply_lut=apply_lut)
    except (ValueError, ZeroDivisionError):
        return np.array([]), np.array([])
    y = data.mean(axis=0)
    x0, _y0 = obj.canvas_to_axes(QPoint(xc0, yc0))
    x1, _y1 = obj.canvas_to_axes(QPoint(xc1, yc1))
    x = np.linspace(x0, x1, len(y))
    return x, y

def get_plot_y_section(obj, apply_lut=False):
    """
    Return plot cross section along y-axis,
    at the x value defined by 'obj', a Marker/AnnotatedPoint object
    """
    x0, _y0 = get_object_coordinates(obj)
    plot = obj.plot()
    ymap = plot.canvasMap(plot.AXES["left"])
    yc0, yc1 = ymap.p1(), ymap.p2()
    if plot.get_axis_direction("left"):
        yc1, yc0 = yc0, yc1
    xc0, _yc0 = obj.axes_to_canvas(x0, 0)
    xc1 = xc0+1
    try:
        data = get_image_data(plot, QPoint(xc0, yc0), QPoint(xc1, yc1),
                              apply_lut=apply_lut)
    except (ValueError, ZeroDivisionError):
        return np.array([]), np.array([])
    y = data.mean(axis=1)
    _x0, y0 = obj.canvas_to_axes(QPoint(xc0, yc0))
    _x1, y1 = obj.canvas_to_axes(QPoint(xc1, yc1))
    x = np.linspace(y0, y1, len(y))
    return x, y


def get_plot_average_x_section(obj, apply_lut=False):
    """
    Return cross section along x-axis, averaged on ROI defined by 'obj'
    'obj' is an AbstractShape object supporting the 'get_rect' method
    (RectangleShape, AnnotatedRectangle, etc.)
    """
    x0, y0, x1, y1 = obj.get_rect()
    xc0, yc0 = obj.axes_to_canvas(x0, y0)
    xc1, yc1 = obj.axes_to_canvas(x1, y1)
    invert = False
    if xc0 > xc1:
        invert = True
        xc1, xc0 = xc0, xc1
    ydir = obj.plot().get_axis_direction("left")
    if (ydir and yc0 > yc1) or (not ydir and yc0 < yc1):
        yc1, yc0 = yc0, yc1
    try:
        data = get_image_data(obj.plot(), QPoint(xc0, yc0), QPoint(xc1, yc1),
                              apply_lut=apply_lut)
    except (ValueError, ZeroDivisionError):
        return np.array([]), np.array([])
    y = data.mean(axis=0)
    if invert:
        y = y[::-1]
    x = np.linspace(x0, x1, len(y))
    return x, y
    
def get_plot_average_y_section(obj, apply_lut=False):
    """
    Return cross section along y-axis, averaged on ROI defined by 'obj'
    'obj' is an AbstractShape object supporting the 'get_rect' method
    (RectangleShape, AnnotatedRectangle, etc.)
    """
    x0, y0, x1, y1 = obj.get_rect()
    xc0, yc0 = obj.axes_to_canvas(x0, y0)
    xc1, yc1 = obj.axes_to_canvas(x1, y1)
    invert = False
    ydir = obj.plot().get_axis_direction("left")
    if (ydir and yc0 > yc1) or (not ydir and yc0 < yc1):
        invert = True
        yc1, yc0 = yc0, yc1
    if xc0 > xc1:
        xc1, xc0 = xc0, xc1
    try:
        data = get_image_data(obj.plot(), QPoint(xc0, yc0), QPoint(xc1, yc1),
                              apply_lut=apply_lut)
    except (ValueError, ZeroDivisionError):
        return np.array([]), np.array([])
    y = data.mean(axis=1)
    x = np.linspace(y0, y1, len(y))
    if invert:
        x = x[::-1]
    return x, y


class XCrossSectionItem(CrossSectionItem):
    """A Qwt item representing x-axis cross section data"""
    _inverted = False
    def get_cross_section(self, obj):
        """Get x-cross section data from source image"""
        source = self.get_source_image()
        rect = get_rectangular_area(obj)
        if rect is None:
            # Object is a marker or an annotated point
            _x0, y0 = get_object_coordinates(obj)
            if self.perimage_mode:
                return source.get_xsection(y0, apply_lut=self.apply_lut)
            else:
                return get_plot_x_section(obj, apply_lut=self.apply_lut)
        else:
            if self.perimage_mode:
                x0, y0, x1, y1 = rect
                return source.get_average_xsection(x0, y0, x1, y1,
                                                   apply_lut=self.apply_lut)
            else:
                return get_plot_average_x_section(obj, apply_lut=self.apply_lut)
            
    def update_scale(self):
        plot = self.plot()
        axis_id = plot.xBottom
        source = self.get_source_image()
        sdiv = source.plot().axisScaleDiv(axis_id)
        plot.setAxisScale(axis_id, sdiv.lowerBound(), sdiv.upperBound())
        plot.replot()

class YCrossSectionItem(CrossSectionItem):
    """A Qwt item representing y-axis cross section data"""
    _inverted = True
    def get_cross_section(self, obj):
        """Get y-cross section data from source image"""
        source = self.get_source_image()
        rect = get_rectangular_area(obj)
        if rect is None:
            # Object is a marker or an annotated point
            x0, _y0 = get_object_coordinates(obj)
            if self.perimage_mode:
                return source.get_ysection(x0, apply_lut=self.apply_lut)
            else:
                return get_plot_y_section(obj, apply_lut=self.apply_lut)
        else:
            if self.perimage_mode:
                x0, y0, x1, y1 = rect
                return source.get_average_ysection(x0, y0, x1, y1,
                                                   apply_lut=self.apply_lut)
            else:
                return get_plot_average_y_section(obj, apply_lut=self.apply_lut)
            
    def update_scale(self):
        plot = self.plot()
        axis_id = plot.yLeft
        source = self.get_source_image()
        sdiv = source.plot().axisScaleDiv(axis_id)
        plot.setAxisScale(axis_id, sdiv.lowerBound(), sdiv.upperBound())
        plot.replot()


class CrossSectionPlot(CurvePlot):
    """Cross section plot"""
    _height = None
    _width = None
    CS_AXIS = None
    Z_AXIS = None
    Z_MAX_MAJOR = 5
    CURVETYPE = None
    def __init__(self, parent=None):
        super(CrossSectionPlot, self).__init__(parent=parent, title="",
                                               section="cross_section")
        self.perimage_mode = True
        self.autoscale_mode = True
        self.apply_lut = False
        
        self.last_obj = None
        self.known_items = {}
        self._shapes = {}
        
        self.curveparam = CurveParam(_("Curve"), icon="curve.png")
        self.curveparam.read_config(CONF, "cross_section", "curve")
        self.curveparam.curvetype = self.CURVETYPE
        
        if self._height is not None:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        if self._width is not None:
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
            
        self.label = make.label(_("Enable a marker"), "C", (0,0), "C")
        self.add_item(self.label)
        
        self.setAxisMaxMajor(self.Z_AXIS, self.Z_MAX_MAJOR)
        self.setAxisMaxMinor(self.Z_AXIS, 0)

    def connect_plot(self, plot):
        if not isinstance(plot, ImagePlot):
            # Connecting only to image plot widgets (allow mixing image and 
            # curve widgets for the same plot manager -- e.g. in pyplot)
            return
        self.connect(plot, SIG_ITEMS_CHANGED, self.items_changed)
        self.connect(plot, SIG_LUT_CHANGED, self.lut_changed)
        self.connect(plot, SIG_ACTIVE_ITEM_CHANGED, self.active_item_changed)
        self.connect(plot, SIG_MARKER_CHANGED, self.marker_changed)
        self.connect(plot, SIG_ANNOTATION_CHANGED, self.shape_changed)
        self.connect(plot, SIG_PLOT_LABELS_CHANGED, self.plot_labels_changed)
        self.connect(plot, SIG_AXIS_DIRECTION_CHANGED, self.axis_dir_changed)
        self.plot_labels_changed(plot)
        for axis_id in plot.AXES:
            self.axis_dir_changed(plot, axis_id)
        self.items_changed(plot)
        
    def register_shape(self, plot, shape, final):
        known_shapes = self._shapes.get(plot, [])
        if shape in known_shapes:
            return
        self._shapes[plot] = known_shapes+[shape]
        param = shape.annotationparam
        param.title = "X/Y"
        param.update_annotation(shape)
        self.update_plot(shape)
        
    def unregister_shape(self, shape):
        for plot in self._shapes:
            shapes = self._shapes[plot]            
            if shape in shapes:
                shapes.pop(shapes.index(shape))
                break
        
    def standard_tools(self, manager):
        manager.add_tool(SelectTool)
        manager.add_tool(BasePlotMenuTool, "item")
        manager.add_tool(BasePlotMenuTool, "axes")
        manager.add_tool(BasePlotMenuTool, "grid")
        manager.add_tool(AntiAliasingTool)
        manager.get_default_tool().activate()
        
    def create_cross_section_item(self):
        raise NotImplementedError

    def items_changed(self, plot):
        self.known_items = {}
        
        # Del all cross section items
        self.del_items(self.get_items(item_type=ICurveItemType))
        
        items = plot.get_items(item_type=ICSImageItemType)
        if not items:
            self.replot()
            return
            
        self.curveparam.shade = min([.3, .8/len(items)])
        for item in items:
            curve = self.create_cross_section_item()
            curve.set_source_image(item)
            self.add_item(curve, z=0)
            self.known_items[item] = curve

    def active_item_changed(self, plot):
        """Active item has just changed"""
        self.shape_changed(plot.get_active_item())

    def plot_labels_changed(self, plot):
        """Plot labels have changed"""
        raise NotImplementedError
        
    def axis_dir_changed(self, plot, axis_id):
        """An axis direction has changed"""
        raise NotImplementedError
        
    def marker_changed(self, marker):
        self.update_plot(marker)

    def is_shape_known(self, shape):
        for shapes in self._shapes.values():
            if shape in shapes:
                return True
        else:
            return False
        
    def shape_changed(self, shape):
        if self.is_shape_known(shape):
            self.update_plot(shape)
            
    def get_last_obj(self):
        if self.last_obj is not None:
            return self.last_obj()
        
    def update_plot(self, obj=None):
        """
        Update cross section curve(s) associated to object *obj*
        
        *obj* may be a marker or a rectangular shape
        (see :py:class:`guiqwt.tools.CrossSectionTool` 
        and :py:class:`guiqwt.tools.AverageCrossSectionTool`)
        
        If obj is None, update the cross sections of the last active object
        """
        if obj is None:
            obj = self.get_last_obj()
            if obj is None:
                return
        else:
            self.last_obj = weakref.ref(obj)
        if obj.plot() is None:
            self.unregister_shape(obj)
            return
        if self.label.isVisible():
            self.label.hide()
        for index, (_item, curve) in enumerate(self.known_items.iteritems()):
            if not self.perimage_mode and index > 0:
                curve.hide()
            else:
                curve.show()
                curve.perimage_mode = self.perimage_mode
                curve.autoscale_mode = self.autoscale_mode
                curve.apply_lut = self.apply_lut
                curve.update_item(obj)
        if self.autoscale_mode:
            self.do_autoscale(replot=True)
                
    def export(self):
        """Export cross-section plot in a text file"""
        items = [item for item in self.get_items(item_type=ICurveItemType)
                 if item.isVisible() and not item.is_empty()]
        if not items:
            QMessageBox.warning(self, _("Export"),
                                _("There is no cross section plot to export."))
            return
        if len(items) > 1:
            items = self.get_selected_items()
        if not items:
            QMessageBox.warning(self, _("Export"),
                                _("Please select a cross section plot."))
            return
        x, y = items[0].get_data()
        data = np.array([x, y]).T
        fname = QFileDialog.getSaveFileName(self, _("Export"),
                                            "", _("Text file")+" (*.txt)")
        if fname:
            try:
                np.savetxt(unicode(fname), data, delimiter=',')
            except RuntimeError, error:
                QMessageBox.critical(self, _("Export"),
                                     _("Unable to export cross section data.")+\
                                     "<br><br>"+_("Error message:")+"<br>"+\
                                     str(error))
        
    def toggle_perimage_mode(self, state):
        self.perimage_mode = state
        self.update_plot()
                    
    def toggle_autoscale(self, state):
        self.autoscale_mode = state
        self.update_plot()
        
    def toggle_apply_lut(self, state):
        self.apply_lut = state
        self.update_plot()
        
    def lut_changed(self, plot):
        if self.apply_lut:
            self.update_plot()


class XCrossSectionPlot(CrossSectionPlot):
    """X-axis cross section plot"""
    _height = 130
    CS_AXIS = CurvePlot.xBottom
    Z_AXIS = CurvePlot.yLeft
    CURVETYPE = "Yfx"
    def sizeHint(self):
        return QSize(self.width(), self._height)
        
    def create_cross_section_item(self):
        return XCrossSectionItem(self.curveparam)

    def plot_labels_changed(self, plot):
        """Plot labels have changed"""
        self.set_axis_title("left", plot.get_axis_title("right"))       
        self.set_axis_title("bottom", plot.get_axis_title("bottom"))
        
    def axis_dir_changed(self, plot, axis_id):
        """An axis direction has changed"""
        if axis_id == plot.xBottom:
            self.set_axis_direction("bottom", plot.get_axis_direction("bottom"))
            self.replot()
        

class YCrossSectionPlot(CrossSectionPlot):
    """Y-axis cross section plot"""
    _width = 140
    CS_AXIS = CurvePlot.yLeft
    Z_AXIS = CurvePlot.xBottom
    Z_MAX_MAJOR = 3
    CURVETYPE = "Xfy"
    def sizeHint(self):
        return QSize(self._width, self.height())
    
    def create_cross_section_item(self):
        return YCrossSectionItem(self.curveparam)

    def plot_labels_changed(self, plot):
        """Plot labels have changed"""
        self.set_axis_title("bottom", plot.get_axis_title("right"))       
        self.set_axis_title("left", plot.get_axis_title("left"))
        
    def axis_dir_changed(self, plot, axis_id):
        """An axis direction has changed"""
        if axis_id == plot.yLeft:
            self.set_axis_direction("left", plot.get_axis_direction("left"))
            self.replot()


class CrossSectionWidget(PanelWidget):
    """Cross section widget"""
    CrossSectionPlotKlass = None
    OTHER_PANEL_ID = None
    
    __implements__ = (IPanel,)

    def __init__(self, parent=None):
        super(CrossSectionWidget, self).__init__(parent)
        
        widget_title = _("Cross section tool")
        widget_icon = "csection.png"
        
        self.manager = None # manager for the associated image plot
        
        self.local_manager = PlotManager(self)
        self.cs_plot = self.CrossSectionPlotKlass(parent)
        
        self.spacer1 = QSpacerItem(0, 0)
        self.spacer2 = QSpacerItem(0, 0)
        
        self.toolbar = toolbar = QToolBar(self)
        if self.CrossSectionPlotKlass is YCrossSectionPlot:
            toolbar.setOrientation(Qt.Horizontal)
            layout = QVBoxLayout()
            layout.addSpacerItem(self.spacer1)
            layout.addWidget(toolbar)
            layout.addWidget(self.cs_plot)
            layout.addSpacerItem(self.spacer2)
        else:
            toolbar.setOrientation(Qt.Vertical)
            layout = QHBoxLayout()
            layout.addWidget(self.cs_plot)
            layout.addWidget(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        self.local_manager.add_plot(self.cs_plot)
        
        self.cs_plot.standard_tools(self.local_manager)
        self.setWindowIcon(get_icon(widget_icon))
        self.setWindowTitle(widget_title)
        
    def set_options(self, peritem=None, applylut=None, autoscale=None):
        assert self.manager is not None, "Panel '%s' must be registered to plot manager before changing options" % self.PANEL_ID
        if peritem is not None:
            self.peritem_ac.setChecked(peritem)
        if applylut is not None:
            self.applylut_ac.setChecked(applylut)
        if autoscale is not None:
            self.autoscale_ac.setChecked(autoscale)
    
    def register_panel(self, manager):
        """Register panel to plot manager"""
        self.manager = manager
        for plot in manager.get_plots():
            self.cs_plot.connect_plot(plot)
        self.setup_actions()
        other = manager.get_panel(self.OTHER_PANEL_ID)
        if other is None:
            add_actions(self.toolbar,
                        (self.peritem_ac, self.applylut_ac, None,
                         self.export_ac, self.autoscale_ac, self.refresh_ac))
        else:
            add_actions(self.toolbar,
                        (other.peritem_ac, other.applylut_ac, None,
                         self.export_ac, other.autoscale_ac, other.refresh_ac))
            self.connect(other.peritem_ac, SIGNAL("toggled(bool)"),
                         self.cs_plot.toggle_perimage_mode)
            self.connect(other.applylut_ac, SIGNAL("toggled(bool)"),
                         self.cs_plot.toggle_apply_lut)
            self.connect(other.autoscale_ac, SIGNAL("toggled(bool)"),
                         self.cs_plot.toggle_autoscale)
            self.connect(other.refresh_ac, SIGNAL("triggered()"),
                         self.cs_plot.update_plot)

    def get_plot(self):
        return self.manager.get_active_plot()
        
    def closeEvent(self, event):
        self.hide()
        event.ignore()
        
    def setup_actions(self):
        self.peritem_ac = create_action(self, _("Per image cross-section"),
                        icon=get_icon('csperimage.png'),
                        toggled=self.cs_plot.toggle_perimage_mode,
                        tip=_("Enable the per-image cross-section mode, "
                              "which works directly on image rows/columns.\n"
                              "That is the fastest method to compute "
                              "cross-section curves but it ignores "
                              "image transformations (e.g. rotation)"))
        self.applylut_ac = create_action(self,
                        _("Apply LUT\n(contrast settings)"),
                        icon=get_icon('csapplylut.png'),
                        toggled=self.cs_plot.toggle_apply_lut,
                        tip=_("Apply LUT (Look-Up Table) contrast settings.\n"
                              "This is the easiest way to compare images "
                              "which have slightly different level ranges.\n\n"
                              "Note: LUT is coded over 1024 levels (0...1023)"))
        self.export_ac = create_action(self, _("Export"),
                                   icon=get_std_icon("DialogSaveButton", 16),
                                   triggered=self.cs_plot.export,
                                   tip=_("Export cross section data"))
        self.autoscale_ac = create_action(self, _("Auto-scale"),
                                   icon=get_icon('csautoscale.png'),
                                   toggled=self.cs_plot.toggle_autoscale)
        self.refresh_ac = create_action(self, _("Refresh"),
                                   icon=get_icon('refresh.png'),
                                   triggered=self.cs_plot.update_plot)

        self.peritem_ac.setChecked(True)
        self.autoscale_ac.setChecked(True)
        self.applylut_ac.setChecked(False)
        
    def register_shape(self, shape, final):
        plot = self.get_plot()
        self.cs_plot.register_shape(plot, shape, final)
        
    def update_plot(self, obj=None):
        """
        Update cross section curve(s) associated to object *obj*
        
        *obj* may be a marker or a rectangular shape
        (see :py:class:`guiqwt.tools.CrossSectionTool` 
        and :py:class:`guiqwt.tools.AverageCrossSectionTool`)
        
        If obj is None, update the cross sections of the last active object
        """
        self.cs_plot.update_plot(obj)

assert_interfaces_valid(CrossSectionWidget)


class XCrossSection(CrossSectionWidget):
    """X-axis cross section widget"""
    PANEL_ID = ID_XCS
    OTHER_PANEL_ID = ID_YCS
    CrossSectionPlotKlass = XCrossSectionPlot

class YCrossSection(CrossSectionWidget):
    """
    Y-axis cross section widget
    parent (QWidget): parent widget
    position (string): "left" or "right"
    """
    PANEL_ID = ID_YCS
    OTHER_PANEL_ID = ID_XCS
    CrossSectionPlotKlass = YCrossSectionPlot
    def __init__(self, parent=None, position="right"):
        CrossSectionWidget.__init__(self, parent)
        self.cs_plot.set_axis_direction("bottom", reverse=position == "left")
