# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.styles
-------------

The `styles` module provides set of parameters (DataSet classes) to 
configure `plot items` and `plot tools`.

.. seealso::
        
    Module :py:mod:`guiqwt.plot`
        Module providing ready-to-use curve and image plotting widgets and 
        dialog boxes
    
    Module :py:mod:`guiqwt.curve`
        Module providing curve-related plot items and plotting widgets
        
    Module :py:mod:`guiqwt.image`
        Module providing image-related plot items and plotting widgets
        
    Module :py:mod:`guiqwt.tools`
        Module providing the `plot tools`
        
Reference
~~~~~~~~~

.. autoclass:: CurveParam
   :members:
   :inherited-members:
.. autoclass:: ErrorBarParam
   :members:
   :inherited-members:
.. autoclass:: GridParam
   :members:
   :inherited-members:
.. autoclass:: ImageParam
   :members:
   :inherited-members:
.. autoclass:: TrImageParam
   :members:
   :inherited-members:
.. autoclass:: ImageFilterParam
   :members:
   :inherited-members:
.. autoclass:: HistogramParam
   :members:
   :inherited-members:
.. autoclass:: Histogram2DParam
   :members:
   :inherited-members:
.. autoclass:: AxesParam
   :members:
   :inherited-members:
.. autoclass:: ImageAxesParam
   :members:
   :inherited-members:
.. autoclass:: LabelParam
   :members:
   :inherited-members:
.. autoclass:: LegendParam
   :members:
   :inherited-members:
.. autoclass:: ShapeParam
   :members:
   :inherited-members:
.. autoclass:: AnnotationParam
   :members:
   :inherited-members:
.. autoclass:: AxesShapeParam
   :members:
   :inherited-members:
.. autoclass:: RangeShapeParam
   :members:
   :inherited-members:
.. autoclass:: MarkerParam
   :members:
   :inherited-members:
.. autoclass:: FontParam
   :members:
   :inherited-members:
.. autoclass:: SymbolParam
   :members:
   :inherited-members:
.. autoclass:: LineStyleParam
   :members:
   :inherited-members:
.. autoclass:: BrushStyleParam
   :members:
   :inherited-members:
.. autoclass:: TextStyleParam
   :members:
   :inherited-members:
"""

import numpy as np

from PyQt4.QtGui import QPen, QBrush, QColor, QFont, QFontDialog, QTransform
from PyQt4.QtCore import Qt, QSize, QPointF
from PyQt4.Qwt5 import QwtPlot, QwtPlotCurve, QwtSymbol, QwtPlotMarker

from guidata.dataset.datatypes import (DataSet, ObjectItem, BeginGroup,
                                       EndGroup, Obj, DataSetGroup,
                                       BeginTabGroup, EndTabGroup,
                                       GetAttrProp, NotProp)
from guidata.dataset.dataitems import (ChoiceItem, BoolItem, FloatItem, IntItem,
                                       ImageChoiceItem, ColorItem, StringItem,
                                       ButtonItem, FloatArrayItem, TextItem)
from guidata.dataset.qtwidgets import DataSetEditLayout
from guidata.dataset.qtitemwidgets import DataSetWidget
from guidata.utils import update_dataset

# Local imports
from guiqwt.config import _
from guiqwt.colormap import get_colormap_list, build_icon_from_cmap_name
from guiqwt.signals import SIG_ITEMS_CHANGED

class ItemParameters(object):
    """Class handling QwtPlotItem-like parameters"""
    MULTISEL_DATASETS = []
    # Customizing tab display order:
    ENDING_PARAMETERS = ("CurveParam", "ErrorBarParam",
                         "ShapeParam", "LabelParam", "LegendParam",
                         "GridParam", "AxesParam")
    
    def __init__(self, multiselection=False):
        self.multiselection = multiselection
        self.paramdict = {}
        self.items = set()
        
    @classmethod
    def register_multiselection(cls, klass, klass_ms):
        """Register a DataSet couple: (DataSet, DataSet_for_MultiSelection)"""
        # Inserting element backwards because classes have to be registered
        # from children to parent (see 'add' method to fully understand why)
        cls.MULTISEL_DATASETS.insert(0, (klass, klass_ms))
        
    def __add(self, key, item, param):
        self.paramdict[key] = param
        self.items.add(item)
        
    def add(self, key, item, param):
        if self.multiselection:
            for klass, klass_ms in self.MULTISEL_DATASETS:
                if isinstance(param, klass):
                    title = param.get_title()
                    if key in self.paramdict and not title.endswith('s'):
                        title += 's'
                    param_ms = klass_ms(title=title,
                                        comment=param.get_comment(),
                                        icon=param.get_icon())
                    update_dataset(param_ms, param)
                    self.__add(key, item, param_ms)
                    return
        self.__add(key, item, param)
        
    def get(self, key):
        from copy import copy
        return copy(self.paramdict.get(key))
    
    def update(self, plot):
        #FIXME: without the following workaround, ImagePlot object aspect ratio
        # is changed when pressing button "Apply"
        from guiqwt.image import ImagePlot
        if isinstance(plot, ImagePlot):
            ratio = plot.get_current_aspect_ratio()
        for item in self.items:
            item.set_item_parameters(self)
        plot.replot()
        if isinstance(plot, ImagePlot):
            plot.set_aspect_ratio(ratio=ratio)
            plot.replot()
        plot.emit(SIG_ITEMS_CHANGED, plot)
    
    def edit(self, plot, title, icon):
        paramdict = self.paramdict.copy()
        ending_parameters = []
        for key in self.ENDING_PARAMETERS:
            if key in paramdict:
                ending_parameters.append(paramdict.pop(key))
        parameters = paramdict.values()+ending_parameters
        dset = DataSetGroup(parameters, title=title.rstrip('.'), icon=icon)
        if dset.edit(parent=plot, apply=lambda dset: self.update(plot)):
            self.update(plot)


LINESTYLES = {"-" : "SolidLine",
              "--": "DashLine",
              ":": "DotLine",
              "-." : "DashDotLine",
              }
COLORS = {
          "r" : "red",
          "g" : "green",
          "b" : "blue",
          "c" : "cyan",
          "m" : "magenta",
          "y" : "yellow",
          "k" : "black",
          "w" : "white",
          "G" : "gray",
          }
MARKERS = {
          "+" : QwtSymbol.Cross,
          "o" : QwtSymbol.Ellipse,
          "*" : QwtSymbol.Star1,
          "." : QwtSymbol(QwtSymbol.Ellipse, QBrush(Qt.black),
                          QPen(Qt.black), QSize(3, 3)),
          "x" : QwtSymbol.XCross,
          "s" : QwtSymbol.Rect,
          "d" : QwtSymbol.Diamond,
          "^" : QwtSymbol.UTriangle,
          "v" : QwtSymbol.DTriangle,
          ">" : QwtSymbol.RTriangle,
          "<" : QwtSymbol.LTriangle,
          "h" : QwtSymbol.Star2,
          }

def style_generator(color_keys="bgrcmykG"):
    """Cycling through curve styles"""
    while True:
        for linestyle in sorted(LINESTYLES.keys()):
            for color in color_keys:
                yield color+linestyle

def update_style_attr(style, param):
    """Parse a MATLAB-like style string and
    update the color, linestyle, marker attributes of the param
    object
    """
    for marker in MARKERS.keys():
        if marker in style:
            param.symbol.update_param(MARKERS[marker])
            break
    else:
        param.symbol.update_param(QwtSymbol.NoSymbol)
    for linestyle in LINESTYLES.keys():
        if linestyle in style:
            param.line.style = LINESTYLES[linestyle]
            break
    else:
        param.line.style = "NoPen"
    for color in COLORS.keys():
        if color in style:
            param.line.color = COLORS[color]
            param.symbol.facecolor = COLORS[color]
            param.symbol.edgecolor = COLORS[color]
            break


def build_reverse_map(lst, obj):
    dict = {}
    for idx, _name, _icon in lst:
        val = getattr(obj, idx)
        dict[val] = idx
    return dict

LINESTYLE_CHOICES = [("SolidLine", _("Solid line"), "solid.png"),
                     ("DashLine", _("Dashed line"), "dash.png"),
                     ("DotLine", _("Dotted line"), "dot.png"),
                     ("DashDotLine", _("Dash-dot line"), "dashdot.png"),
                     ("DashDotDotLine", _("Dash-dot-dot line"), "dashdotdot.png"),
                     ("NoPen", _("No line"), "none.png"),
                     ]
MARKER_CHOICES = [("Cross", _("Cross"), "cross.png"),
                  ("Ellipse", _("Ellipse"), "ellipse.png"),
                  ("Star1", _("Star"), "star.png"),
                  ("XCross", _("X-Cross"), "xcross.png"),
                  ("Rect", _("Square"), "square.png"),
                  ("Diamond", _("Diamond"), "diamond.png"),
                  ("UTriangle", _("Triangle"), "triangle_u.png"),
                  ("DTriangle", _("Triangle"), "triangle_d.png"),
                  ("RTriangle", _("Triangle"), "triangle_r.png"),
                  ("LTriangle", _("Triangle"), "triangle_l.png"),
                  ("Star2", _("Hexagon"), "hexagon.png"),
                  ("NoSymbol", _("No symbol"), "none.png"),
                  ]
CURVESTYLE_CHOICES = [("Lines", _("Lines"), "lines.png"),
                      ("Sticks", _("Sticks"), "sticks.png"),
                      ("Steps", _("Steps"), "steps.png"),
                      ("Dots", _("Dots"), "dots.png"),
                      ("NoCurve", _("No curve"), "none.png")
                      ]
CURVETYPE_CHOICES = [("Yfx", _("Draws y as a function of x"), "yfx.png"),
                     ("Xfy", _("Draws x as a function of y"), "xfy.png")]

BRUSHSTYLE_CHOICES = [
    ("NoBrush", _("No brush pattern"), "nobrush.png"),
    ("SolidPattern", _("Uniform color"), "solidpattern.png"),
    ("Dense1Pattern", _("Extremely dense brush pattern"), "dense1pattern.png"),
    ("Dense2Pattern", _("Very dense brush pattern"), "dense2pattern.png"),
    ("Dense3Pattern", _("Somewhat dense brush pattern"), "dense3pattern.png"),
    ("Dense4Pattern", _("Half dense brush pattern"), "dense4pattern.png"),
    ("Dense5Pattern", _("Somewhat sparse brush pattern"), "dense5pattern.png"),
    ("Dense6Pattern", _("Very sparse brush pattern"), "dense6pattern.png"),
    ("Dense7Pattern", _("Extremely sparse brush pattern"), "dense7pattern.png"),
    ("HorPattern", _("Horizontal lines"), "horpattern.png"),
    ("VerPattern", _("Vertical lines"), "verpattern.png"),
    ("CrossPattern", _("Crossing horizontal and vertical lines"),
     "crosspattern.png"),
    ("BDiagPattern", _("Backward diagonal lines"), "bdiagpattern.png"),
    ("FDiagPattern", _("Forward diagonal lines"), "fdiagpattern.png"),
    ("DiagCrossPattern", _("Crossing diagonal lines"), "diagcrosspattern.png"),
#    ("LinearGradientPattern", _("Linear gradient (set using a dedicated QBrush constructor)"), "none.png"),
#    ("ConicalGradientPattern", _("Conical gradient (set using a dedicated QBrush constructor)"), "none.png"),
#    ("RadialGradientPattern", _("Radial gradient (set using a dedicated QBrush constructor)"), "none.png"),
#    ("TexturePattern", _("Custom pattern (see QBrush::setTexture())"), "none.png"),
]

MARKER_NAME = build_reverse_map(MARKER_CHOICES, QwtSymbol)
#for _name in dir(QwtSymbol):
#    _val = getattr(QwtSymbol, _name)
#    if isinstance(_val, QwtSymbol.Style):
#        MARKER_NAME[_val] = _name

CURVESTYLE_NAME = build_reverse_map(CURVESTYLE_CHOICES, QwtPlotCurve)

CURVETYPE_NAME = build_reverse_map(CURVETYPE_CHOICES, QwtPlotCurve)

LINESTYLE_NAME = build_reverse_map(LINESTYLE_CHOICES, Qt)
#for _name in dir(Qt):
#    _val = getattr(Qt, _name)
#    if isinstance(_val, Qt.PenStyle):
#        LINESTYLE_NAME[_val] = _name
BRUSHSTYLE_NAME = build_reverse_map(BRUSHSTYLE_CHOICES, Qt)

# ===================================================
# Common font parameters
# ===================================================
def _font_selection(param, item, value):
    font = param.build_font()
    result, valid = QFontDialog.getFont( font )
    if valid:
        param.update_param( result )
    
class FontParam(DataSet):
    family = StringItem(_("Family"), default="default")
    _choose = ButtonItem(_("Choose font"), _font_selection,
                         default=None).set_pos(col=1)
    size = IntItem(_("Size in point"), default=12)
    bold = BoolItem(_("Bold"), default=False).set_pos(col=1)
    italic = BoolItem(_("Italic"), default=False).set_pos(col=2)

    def update_param(self, font):
        self.family = str(font.family())
        self.size = font.pointSize()
        self.bold = bool(font.bold())
        self.italic = bool(font.italic())

    def build_font(self):
        font = QFont(self.family)
        font.setPointSize( self.size )
        font.setBold( self.bold )
        font.setItalic( self.italic )
        return font

class FontItemWidget(DataSetWidget):
    klass = FontParam

class FontItem(ObjectItem):
    """Item holding a LineStyleParam"""
    klass = FontParam

DataSetEditLayout.register(FontItem, FontItemWidget)


# ===================================================
# Common Qwt symbol parameters
# ===================================================
class SymbolParam(DataSet):
    marker = ImageChoiceItem(_("Style"), MARKER_CHOICES,
                             default="NoSymbol")
    size = IntItem(_("Size"), default=9)
    edgecolor = ColorItem(_("Border"), default="gray")
    facecolor = ColorItem(_("Background color"), default="yellow")
    alpha = FloatItem(_("Background alpha"), default=1., min=0, max=1)

    def update_param(self, symb):
        if not isinstance(symb, QwtSymbol):
            # check if this is still needed
            #raise RuntimeError
            assert isinstance(symb, QwtSymbol.Style)
            self.marker = MARKER_NAME[symb]
            return
        self.marker = MARKER_NAME[symb.style()]
        self.size = symb.size().width()
        self.edgecolor = str(symb.pen().color().name())
        self.facecolor = str(symb.brush().color().name())

    def build_symbol(self):
        marker_type = getattr(QwtSymbol, self.marker)
        color = QColor(self.facecolor)
        color.setAlphaF(self.alpha)
        marker = QwtSymbol(marker_type, QBrush(color),
                           QPen(QColor(self.edgecolor)),
                           QSize(self.size, self.size))
        return marker
    
    def update_symbol(self, obj):
        obj.setSymbol(self.build_symbol())        

class SymbolItemWidget(DataSetWidget):
    klass = SymbolParam

class SymbolItem(ObjectItem):
    """Item holding a SymbolParam"""
    klass = SymbolParam

DataSetEditLayout.register(SymbolItem, SymbolItemWidget)


# ===================================================
# Common line style parameters
# ===================================================
class LineStyleParam(DataSet):
    style = ImageChoiceItem(_("Style"), LINESTYLE_CHOICES, default="SolidLine")
    color = ColorItem(_("Color"), default="black")
    width = FloatItem(_("Width"), default=1., min=0)

    def update_param(self, pen):
        self.width = pen.widthF()
        self.color = str(pen.color().name())
        self.style = LINESTYLE_NAME[pen.style()]

    def build_pen(self):
        linecolor = QColor(self.color)
        style = getattr(Qt, self.style)
        pen = QPen(linecolor, self.width, style)
        return pen
        
    def set_style_from_matlab(self, linestyle):
        """Eventually convert MATLAB-like linestyle into Qt linestyle"""
        linestyle = LINESTYLES.get(linestyle, linestyle) # MATLAB-style
        if linestyle == '': # MATLAB-style
            linestyle = 'NoPen'
        self.style = linestyle

class LineStyleItemWidget(DataSetWidget):
    klass = LineStyleParam

class LineStyleItem(ObjectItem):
    """Item holding a LineStyleParam"""
    klass = LineStyleParam

DataSetEditLayout.register(LineStyleItem, LineStyleItemWidget)

# ===================================================
# Common brush style parameters
# ===================================================
class BrushStyleParam(DataSet):
    style = ImageChoiceItem(_("Style"), BRUSHSTYLE_CHOICES,
                            default="SolidPattern")
    color = ColorItem(_("Color"), default="black")
    alpha = FloatItem(_("Alpha"), default=1.0)
    angle = FloatItem(_("Angle"), default=0., min=0)
    sx = FloatItem(_("sx"), default=1., min=0)
    sy = FloatItem(_("sy"), default=1., min=0)

    def update_param(self, brush):
        from math import pi, sqrt, atan2
        tr = brush.transform()
        pt = tr.map( QPointF(1.0, 0.0) )
        self.sx = sqrt(pt.x()**2+pt.y()**2)
        self.angle = 180*atan2(pt.y(), pt.x())/pi
        pt = tr.map( QPointF(0.0, 1.0) )
        self.sy = sqrt(pt.x()**2+pt.y()**2)

        col = brush.color()
        self.color = str(col.name())
        self.alpha = col.alphaF()
        self.style = BRUSHSTYLE_NAME[brush.style()]

    def build_brush(self):
        color = QColor(self.color)
        color.setAlphaF(self.alpha)
        brush = QBrush(color, getattr(Qt,self.style))
        tr = QTransform()
        tr = tr.scale(self.sx, self.sy)
        tr = tr.rotate(self.angle)
        brush.setTransform(tr)
        return brush

class BrushStyleItemWidget(DataSetWidget):
    klass = BrushStyleParam

class BrushStyleItem(ObjectItem):
    """Item holding a LineStyleParam"""
    klass = BrushStyleParam

DataSetEditLayout.register(BrushStyleItem, BrushStyleItemWidget)


# ===================================================
# QwtText parameters
# ===================================================
class TextStyleParam(DataSet):
    font = FontItem(_("Font"))
    textcolor = ColorItem(_("Text color"), default="blue")
    background_color = ColorItem(_("Background color"), default="white")
    background_alpha = FloatItem(_("Background alpha"),
                            default=0.5, min=0, max=1)

    def update_param(self, obj):
        """obj: QwtText instance"""
        self.font.update_param( obj.font() )
        self.textcolor = obj.color().name()
        color = obj.backgroundBrush().color()
        self.background_color = color.name()
        self.background_alpha = color.alphaF()
    
    def update_text(self, obj):
        """obj: QwtText instance"""
        obj.setColor( QColor(self.textcolor) )
        color = QColor(self.background_color)
        color.setAlphaF(self.background_alpha)
        obj.setBackgroundBrush( QBrush(color) )
        font = self.font.build_font()
        obj.setFont(font)

class TextStyleItemWidget(DataSetWidget):
    klass = TextStyleParam

class TextStyleItem(ObjectItem):
    """Item holding a TextStyleParam"""
    klass = TextStyleParam

DataSetEditLayout.register(TextStyleItem, TextStyleItemWidget)


# ===================================================
# Grid parameters
# ===================================================
class GridParam(DataSet):
    background = ColorItem(_("Background color"), default="white")
    maj = BeginGroup(_("Major grid") )
    maj_xenabled = BoolItem(_("X Axis"), default=True)
    maj_yenabled = BoolItem(_("Y Axis"), default=True).set_pos(col=1)
    maj_line = LineStyleItem(_("Line"))
    _maj = EndGroup("end group")
    
    min = BeginGroup(_("Minor grid"))    
    min_xenabled = BoolItem(_("X Axis"), default=False)
    min_yenabled = BoolItem(_("Y Axis"), default=False).set_pos(col=1)
    min_line = LineStyleItem(_("Line"))
    _min = EndGroup("fin groupe")

    def update_param(self, grid):
        plot = grid.plot()
        if plot is not None:
            self.background = str(plot.canvasBackground().name())
        self.maj_xenabled = grid.xEnabled()
        self.maj_yenabled = grid.yEnabled()
        self.maj_line.update_param( grid.majPen() )
        self.min_xenabled = grid.xMinEnabled()
        self.min_yenabled = grid.yMinEnabled()
        self.min_line.update_param( grid.minPen() )

    def update_grid(self, grid):
        plot = grid.plot()
        if plot is not None:
            plot.setCanvasBackground( QColor(self.background) )
        grid.enableX(self.maj_xenabled)
        grid.enableY(self.maj_yenabled)
        grid.setPen( self.maj_line.build_pen() )
        grid.enableXMin(self.min_xenabled)
        grid.enableYMin(self.min_yenabled)
        grid.setMinPen( self.min_line.build_pen() )
        grid.setTitle(self.get_title())


# ===================================================
# Axes style parameters
# ===================================================
class AxeStyleParam(DataSet):
    title = StringItem(_("Title"), default=u"")
    color = ColorItem(_("Color"), default="black").set_pos(col=1)
    title_font = FontItem(_("Title font"))
    ticks_font = FontItem(_("Values font"))

    
# ===================================================
# Axes parameters
# ===================================================
class AxesParam(DataSet):
    xparams = BeginGroup(_("X Axis") )
    xaxis = ChoiceItem(_("Position"),
                       [(QwtPlot.xBottom, _("bottom")),
                        (QwtPlot.xTop, _("top"))],
                       default=QwtPlot.xBottom, help=_("X-axis position"))
    xscale = ChoiceItem(_("Scale"),
                        [("lin", _("linear")), ("log", _("logarithmic"))],
                        default="lin")
    xmin = FloatItem("x|min", help=_("Lower x-axis limit"))
    xmax = FloatItem("x|max", help=_("Upper x-axis limit"))
    _xparams = EndGroup("end X")
    yparams = BeginGroup(_("Y Axis") )
    yaxis = ChoiceItem(_("Position"),
                       [(QwtPlot.yLeft, _("left")),
                        (QwtPlot.yRight, _("right"))],
                       default=QwtPlot.yLeft, help=_("Y-axis position"))
    yscale = ChoiceItem(_("Scale"),
                        [("lin", _("linear")), ("log", _("logarithmic"))],
                        default="lin")
    ymin = FloatItem("y|min", help=_("Lower y-axis limit"))
    ymax = FloatItem("y|max", help=_("Upper y-axis limit"))
    _yparams = EndGroup("end Y")

    def update_param(self, item):
        sw = item.plot()
        self.xaxis = item.xAxis()
        self.xscale = sw.get_axis_scale(self.xaxis)
        xaxis = sw.axisScaleDiv(self.xaxis)
        self.xmin = xaxis.lowerBound()
        self.xmax = xaxis.upperBound()
        self.yaxis = item.yAxis()
        self.yscale = sw.get_axis_scale(self.yaxis)
        yaxis = sw.axisScaleDiv(self.yaxis)
        self.ymin = yaxis.lowerBound()
        self.ymax = yaxis.upperBound()

    def update_axes(self, item):
        sw = item.plot()
        sw.grid.setAxis(self.xaxis, self.yaxis)
        # x
        item.setXAxis(self.xaxis)
        sw.enableAxis(self.xaxis, True)
        sw.set_axis_scale(self.xaxis, self.xscale)
        sw.setAxisScale(self.xaxis, self.xmin, self.xmax)
        # y
        item.setYAxis(self.yaxis)
        sw.enableAxis(self.yaxis, True)
        sw.set_axis_scale(self.yaxis, self.yscale)
        sw.setAxisScale(self.yaxis, self.ymin, self.ymax)
        
        sw.disable_unused_axes()

class ImageAxesParam(DataSet):
    xparams = BeginGroup(_("X Axis") )
    xmin = FloatItem("x|min", help=_("Lower x-axis limit"))
    xmax = FloatItem("x|max", help=_("Upper x-axis limit"))
    _xparams = EndGroup("end X")
    yparams = BeginGroup(_("Y Axis") )
    ymin = FloatItem("y|min", help=_("Lower y-axis limit"))
    ymax = FloatItem("y|max", help=_("Upper y-axis limit"))
    _yparams = EndGroup("end Y")
    zparams = BeginGroup(_("Z Axis") )
    zmin = FloatItem("z|min", help=_("Lower z-axis limit"))
    zmax = FloatItem("z|max", help=_("Upper z-axis limit"))
    _zparams = EndGroup("end Z")

    def update_param(self, item):
        sw = item.plot()
        xaxis = sw.axisScaleDiv(item.xAxis())
        self.xmin = xaxis.lowerBound()
        self.xmax = xaxis.upperBound()
        yaxis = sw.axisScaleDiv(item.yAxis())
        self.ymin = yaxis.lowerBound()
        self.ymax = yaxis.upperBound()
        self.zmin, self.zmax = item.min, item.max

    def update_axes(self, item):
        sw = item.plot()
        sw.setAxisScale(item.xAxis(), self.xmin, self.xmax)
        sw.setAxisScale(item.yAxis(), self.ymin, self.ymax)
        item.set_lut_range([self.zmin, self.zmax])
        sw.update_colormap_axis(item)

# ===================================================
# Marker parameters
# ===================================================
MARKER_LINE_STYLE = [ QwtPlotMarker.NoLine,
                      QwtPlotMarker.HLine,
                      QwtPlotMarker.VLine,
                      QwtPlotMarker.Cross ]
class MarkerParam(DataSet):
    symbol = SymbolItem(_("Symbol"))
    text = TextStyleItem(_("Text"))
    pen = LineStyleItem(_("Line"))
    linestyle = ChoiceItem(_("Line style"),
                       [(0, _("None")),
                        (1, _("Horizontal")),
                        (2, _("Vertical")),
                        (3, _("Both")),
                        ],  default=0)
    spacing = IntItem(_("Spacing"), default=10, min=0)
    
    def update_param(self, obj):
        self.symbol.update_param(obj.symbol())
        self.text.update_param(obj.label())
        self.pen.update_param(obj.linePen())
        ls = obj.lineStyle()
        self.linestyle = MARKER_LINE_STYLE.index( ls )
        self.spacing = obj.spacing()

    def update_marker(self, obj):
        self.symbol.update_symbol(obj)
        label = obj.label()
        self.text.update_text(label)
        obj.setLabel(label)
        pen = self.pen.build_pen()
        obj.setLinePen(pen)
        obj.setLineStyle( MARKER_LINE_STYLE[self.linestyle] )
        obj.setSpacing(self.spacing)


# ===================================================
# Label parameters
# ===================================================

class LabelParam(DataSet):
    _multiselection = False
    _legend = False
    _no_contents = True
    label = StringItem(_("Title"), default="") \
            .set_prop("display", hide=GetAttrProp("_multiselection"))
            
    _styles = BeginTabGroup("Styles")
    #-------------------------------------------------------------- Contents tab
    ___cont = BeginGroup(_("Contents")).set_prop("display", icon="label.png",
                                             hide=GetAttrProp("_no_contents"))
    contents = TextItem("").set_prop("display",
                                     hide=GetAttrProp("_no_contents"))
    ___econt = EndGroup(_("Contents")).set_prop("display",
                                             hide=GetAttrProp("_no_contents"))
    #---------------------------------------------------------------- Symbol tab
    symbol = SymbolItem(_("Symbol")).set_prop("display", icon="diamond.png",
                                              hide=GetAttrProp("_legend"))
    #---------------------------------------------------------------- Border tab
    border = LineStyleItem(_("Border"), default=Obj(color="#cbcbcb"),
                           help=_("set width to 0 to disable")
                           ).set_prop("display", icon="dashdot.png")
    #------------------------------------------------------------------ Text tab
    ___text = BeginGroup(_("Text")).set_prop("display", icon="font.png")
    font = FontItem(_("Text font"))
    color = ColorItem(_("Text color"), default="#000000")
    bgcolor = ColorItem(_("Background color"), default="#ffffff")
    bgalpha = FloatItem(_("Background transparency"),
                        min=0.0, max=1.0, default=0.8)
    ___etext = EndGroup(_("Text"))
    #-------------------------------------------------------------- Position tab
    ___position = BeginGroup(_("Position")).set_prop("display", icon="move.png")
    _begin_anchor = BeginGroup(_("Position relative to anchor")) \
                    .set_prop("display", hide=GetAttrProp("_multiselection"))
    anchor = ChoiceItem(_("Corner"),
                        [("TL" , _("Top left") ),
                         ("TR" , _("Top right") ),
                         ("BL" , _("Bottom left") ),
                         ("BR" , _("Bottom right") ),
                         ("L"  , _("Left") ),
                         ("R"  , _("Right") ),
                         ("T"  , _("Top") ),
                         ("B"  , _("Bottom") ),
                         ("C"  , _("Center") ),], default="TL",
                         help=_("Label position relative to anchor point")) \
                         .set_prop("display",
                                   hide=GetAttrProp("_multiselection"))
    xc = IntItem(_(u"ΔX"), default=5,
                 help=_("Horizontal offset (pixels) relative to anchor point"))\
                 .set_prop("display", hide=GetAttrProp("_multiselection"))
    yc = IntItem(_(u"ΔY"), default=5,
                 help=_("Vertical offset (pixels) relative to anchor point")
                 ).set_pos(col=1).set_prop("display",
                                           hide=GetAttrProp("_multiselection"))
    _end_anchor = EndGroup(_("Anchor")) \
                  .set_prop("display", hide=GetAttrProp("_multiselection"))
    _begin_anchorpos = BeginGroup(_("Anchor position")) \
                       .set_prop("display", hide=GetAttrProp("_multiselection"))
    _abspos_prop = GetAttrProp("abspos")
    abspos = BoolItem(text=_("Attach to canvas"), label=_("Anchor"),
                      default=True
                      ).set_prop("display", store=_abspos_prop) \
                       .set_prop("display", hide=GetAttrProp("_multiselection"))
    xg = FloatItem(_("X"), default=0.0,
                   help=_("X-axis position in canvas coordinates")
                   ).set_prop("display", active=NotProp(_abspos_prop)) \
                    .set_prop("display", hide=GetAttrProp("_multiselection"))
    yg = FloatItem(_("Y"), default=0.0,
                   help=_("Y-axis position in canvas coordinates")
                   ).set_pos(col=1) \
                    .set_prop("display", active=NotProp(_abspos_prop)) \
                    .set_prop("display", hide=GetAttrProp("_multiselection"))
    move_anchor = ChoiceItem(_(u"Interact"),
                         ((True, _("moving object changes anchor position")),
                          (False, _("moving object changes label position"))),
                         default=True
                         ).set_prop("display", active=NotProp(_abspos_prop)) \
                          .set_prop("display",
                                    hide=GetAttrProp("_multiselection"))
    absg = ChoiceItem(_("Position"),
                        [("TL", _("Top left") ),
                         ("TR", _("Top right") ),
                         ("BL", _("Bottom left") ),
                         ("BR", _("Bottom right") ),
                         ("L" , _("Left") ),
                         ("R" , _("Right") ),
                         ("T" , _("Top") ),
                         ("B" , _("Bottom") ),
                         ("C" , _("Center") ),], default="TL",
                         help=_("Absolute position on canvas")
                         ).set_prop("display", active=_abspos_prop) \
                          .set_prop("display",
                                    hide=GetAttrProp("_multiselection"))
    _end_anchorpos = EndGroup(_("Anchor position")) \
                     .set_prop("display", hide=GetAttrProp("_multiselection"))
    ___eposition = EndGroup(_("Position"))
    #----------------------------------------------------------------------- End
    _endstyles = EndTabGroup("Styles")

    def update_param(self, obj):
        # The following is necessary only for shape labels:
        # when shape is just created (and not yet moved), we need to update
        # these attributes
        if self.abspos:
            self.absg = obj.G
        else:
            self.xg, self.yg = obj.G
        self.xc, self.yc = obj.C

    def update_label(self, obj):
        if not self._multiselection:
            if self.abspos:
                obj.G = self.absg
            else:
                obj.G = (self.xg, self.yg)
            obj.C = self.xc, self.yc
            obj.anchor = self.anchor
            obj.move_anchor = self.move_anchor
            obj.setTitle(self.label)
        obj.marker = self.symbol.build_symbol()
        obj.border_pen = self.border.build_pen()
        obj.set_text_style(self.font.build_font(), self.color)
        color = QColor(self.bgcolor)
        color.setAlphaF(self.bgalpha)
        obj.bg_brush = QBrush(color)

class LabelParam_MS(LabelParam):
    _multiselection = True
    
ItemParameters.register_multiselection(LabelParam, LabelParam_MS)

class LegendParam(LabelParam):
    _legend = True
    label = StringItem(_("Title"), default="").set_prop("display", hide=True)
    
    def update_label(self, obj):
        super(LegendParam, self).update_label(obj)
        if not self._multiselection:
            obj.setTitle(self.get_title())

class LegendParam_MS(LegendParam):
    _multiselection = True
    
ItemParameters.register_multiselection(LegendParam, LegendParam_MS)

class LabelParamWithContents(LabelParam):
    _no_contents = False
    def __init__(self, title=None, comment=None, icon=''):
        self.plain_text = None
        super(LabelParamWithContents, self).__init__(title, comment, icon)
        
    def update_param(self, obj):
        super(LabelParamWithContents, self).update_param(obj)
        self.contents = self.plain_text = obj.get_plain_text()

    def update_label(self, obj):
        super(LabelParamWithContents, self).update_label(obj)
        if self.plain_text is not None and self.contents != self.plain_text:
            text = self.contents.replace('\n', '<br>')
            obj.set_text(text)

class LabelParamWithContents_MS(LabelParamWithContents):
    _multiselection = True
    
ItemParameters.register_multiselection(LabelParamWithContents,
                                       LabelParamWithContents_MS)


# ===================================================
# Curve parameters
# ===================================================
class CurveParam(DataSet):
    _multiselection = False
    label = StringItem(_("Title"), default="").set_prop("display",
                                          hide=GetAttrProp("_multiselection"))
    line = LineStyleItem(_("Line"))
    symbol = SymbolItem(_("Symbol"))
    shade = FloatItem(_("Shadow"), default=0, min=0, max=1)
    fitted = BoolItem(_("Fit curve to data"), _("Fitting"), default=False)
    curvestyle = ImageChoiceItem(_("Curve style"), CURVESTYLE_CHOICES,
                                 default="Lines")
    curvetype = ImageChoiceItem(_("Curve type"), CURVETYPE_CHOICES,
                                default="Yfx")
    baseline = FloatItem(_("Baseline"), default=0.)

    def update_param(self, curve):
        self.symbol.update_param(curve.symbol())
        self.line.update_param(curve.pen())
        self.curvestyle = CURVESTYLE_NAME[curve.style()]
        self.curvetype = CURVETYPE_NAME[curve.curveType()]
        self.baseline = curve.baseline()
    
    def update_curve(self, curve):
        if not self._multiselection:
            # Non common parameters
            curve.setTitle(self.label)
        pen = self.line.build_pen()
        curve.setPen(pen)
        # Brush
        linecolor = QColor(self.line.color)
        linecolor.setAlphaF(self.shade)
        brush = QBrush(linecolor)
        if not self.shade:
            brush.setStyle(Qt.NoBrush)
        curve.setBrush(brush)
        # Symbol
        self.symbol.update_symbol( curve )
        # CurveAttribute
        curve.setCurveAttribute(QwtPlotCurve.Fitted, self.fitted)
        # Curve style, type and baseline
        curve.setStyle(getattr(QwtPlotCurve, self.curvestyle))
        curve.setCurveType(getattr(QwtPlotCurve, self.curvetype))
        curve.setBaseline(self.baseline)

class CurveParam_MS(CurveParam):
    _multiselection = True
    
ItemParameters.register_multiselection(CurveParam, CurveParam_MS)


# ===================================================
# ErrorBar Curve parameters
# ===================================================
class ErrorBarParam(DataSet):
    mode = ChoiceItem(_("Display"), default=0,
                      choices=[_("error bars with caps (x, y)"),
                               _("error area (y)")],
                      help=_("Note: only y-axis error bars are shown in "
                             "error area mode\n(width and cap parameters "
                             "will also be ignored)"))
    color = ColorItem(_("Color"), default="darkred")
    alpha = FloatItem(_("Alpha"), default=.9, min=0, max=1,
                      help=_("Error bar transparency"))
    width = FloatItem(_("Width"), default=1.0, min=1)
    cap = IntItem(_("Cap"), default=4, min=0)
    ontop = BoolItem(_("set to foreground"), _("Visibility"), default=False)

    def update_param(self, curve):
        color = curve.errorPen.pen().color()
        self.color = str(color.name())
        self.alpha = color.alphaF()
        self.width = curve.errorPen.widthF()
        self.cap = curve.errorCap
        self.ontop = curve.errorOnTop

    def update_curve(self, curve):
        color = QColor(self.color)
        color.setAlphaF(self.alpha)
        curve.errorPen = QPen(color, self.width)
        curve.errorBrush = QBrush(color)
        curve.errorCap = self.cap
        curve.errorOnTop = self.ontop


# ===================================================
# Image parameters
# ===================================================
def _create_choices():
    choices = []
    for cmap_name in get_colormap_list():
        choices.append((cmap_name, cmap_name, build_icon_from_cmap_name))
    return choices


class ImageParam(DataSet):
    _multiselection = False
    label = StringItem(_("Image title"), default=_("Image")) \
            .set_prop("display", hide=GetAttrProp("_multiselection"))
    _hide_background = False
    background = ColorItem(_("Background color"), default="#000000"
                           ).set_prop("display",
                                      hide=GetAttrProp("_hide_background"))
    alpha_mask = BoolItem(_("Use image level as alpha"), _("Alpha channel"),
                          default=False)
    alpha = FloatItem(_("Global alpha"), default=1.0,
                      help=_("Global alpha value"))
    colormap = ImageChoiceItem(_("Colormap"), _create_choices(), default="jet")
    
    interpolation = ChoiceItem(_("Interpolation"),
                               [(0, _("None (nearest pixel)")),
                                (1, _("Linear interpolation")),
                                (2, _("2x2 antialiasing filter")),
                                (3, _("3x3 antialiasing filter")),
                                (5, _("5x5 antialiasing filter"))],
                               default=0, help=_("Image interpolation type"))
                               
    def update_param(self, image):
        self.label = unicode(image.title().text())
        self.background = QColor(image.bg_qcolor).name()
        self.colormap = image.get_color_map_name()
        interpolation = image.get_interpolation()
        mode = interpolation[0]
        from guiqwt.image import INTERP_NEAREST, INTERP_LINEAR
        if mode == INTERP_NEAREST:
            self.interpolation = 0
        elif mode == INTERP_LINEAR:
            self.interpolation = 1
        else:
            size = interpolation[1].shape[0]
            self.interpolation = size

    def update_image(self, image):
        image.setTitle(self.label)
        image.set_background_color(self.background)
        image.set_color_map(self.colormap)
        size = self.interpolation
        from guiqwt.image import INTERP_NEAREST, INTERP_LINEAR, INTERP_AA
        if size == 0:
            mode = INTERP_NEAREST
        elif size == 1:
            mode = INTERP_LINEAR
        else:
            mode = INTERP_AA
        image.set_interpolation(mode, size)

class ImageParam_MS(ImageParam):
    _multiselection = True
    
ItemParameters.register_multiselection(ImageParam, ImageParam_MS)


class ImageFilterParam(ImageParam):
    label = StringItem(_("Title"), default=_("Filter"))
    g1 = BeginGroup(_("Bounds"))
    xmin = FloatItem(_("x|min"))
    xmax = FloatItem(_("x|max"))
    ymin = FloatItem(_("y|min"))
    ymax = FloatItem(_("y|max"))
    _g1 = EndGroup("sub-group")
    use_source_cmap = BoolItem(_("Use image colormap and level"),
                               _("Color map"), default=True)
    
    def update_param(self, obj):
        self.xmin, self.ymin, self.xmax, self.ymax = obj.border_rect.get_rect()
        self.use_source_cmap = obj.use_source_cmap
        super(ImageFilterParam,self).update_param(obj)
    
    def update_imagefilter(self, imagefilter):
        m, M = imagefilter.get_lut_range()
        set_range = False
        if not self.use_source_cmap and imagefilter.use_source_cmap:
            set_range = True
        imagefilter.use_source_cmap = self.use_source_cmap
        if set_range:
            imagefilter.set_lut_range([m, M])
        self.update_image(imagefilter)
        imagefilter.border_rect.set_rect(self.xmin, self.ymin,
                                         self.xmax, self.ymax)


class TrImageParam(ImageParam):
    _crop = BeginGroup(_("Crop"))
    crop_left = IntItem(_("Left"), default=0)
    crop_right = IntItem(_("Right"), default=0)
    crop_top = IntItem(_("Top"), default=0)
    crop_bottom = IntItem(_("Bottom"), default=0)
    _end_crop = EndGroup(_("Cropping"))
    _ps = BeginGroup(_("Pixel size"))
    dx = FloatItem(_("Width (δx)"), default=1.0)
    dy = FloatItem(_("Height (δy)"), default=1.0)
    _end_ps = EndGroup(_("Pixel size"))
    _pos = BeginGroup(_("Translate, rotate and flip"))
    pos_x0 = FloatItem(_("x<sub>CENTER</sub>"), default=0.0)
    hflip = BoolItem(_("Flip horizontally"), default=False).set_prop("display", col=1)
    pos_y0 = FloatItem(_("y<sub>CENTER</sub>"), default=0.0)
    vflip = BoolItem(_("Flip vertically"), default=False).set_prop("display", col=1)
    pos_angle = FloatItem(_("θ (°)"), default=0.0).set_prop("display", col=0)
    _end_pos = EndGroup(_("Translate, rotate and flip"))

    def update_param(self, image):
        super(TrImageParam, self).update_param(image)
        # we don't get crop info from the image because
        # its not easy to extract from the transform
        # and TrImageItem keeps it's crop information
        # directly in this DataSet

    def update_image(self, image):
        ImageParam.update_image(self, image)
        image.set_transform(*self.get_transform())

    def get_transform(self):
        return (self.pos_x0, self.pos_y0, self.pos_angle*np.pi/180,
                self.dx, self.dy, self.hflip, self.vflip)

    def set_transform(self, x0, y0, angle, dx=1.0, dy=1.0,
                      hflip=False, vflip=False):
        self.pos_x0 = x0
        self.pos_y0 = y0
        self.pos_angle = angle*180/np.pi
        self.dx = dx
        self.dy = dy
        self.hflip = hflip
        self.vflip = vflip

    def set_crop(self, left, top, right, bottom):
        self.crop_left = left
        self.crop_right = right
        self.crop_top = top
        self.crop_bottom = bottom

    def get_crop(self):
        return (self.crop_left, self.crop_top,
                self.crop_right, self.crop_bottom)
    
ItemParameters.register_multiselection(TrImageParam, ImageParam_MS)


# ===================================================
# Histogram parameters
# ===================================================
class HistogramParam(DataSet):
    n_bins = IntItem(_("Bins"), default=100, min=1, help=_("Number of bins"))
    logscale = BoolItem(_("logarithmic"), _("Y-axis scale"), default=False)
    remove_first_bin = BoolItem(_("force first bin to zero"), _("Display"),
                                default=False)

    def update_param(self, obj):
        self.n_bins = obj.get_bins()
        self.logscale = obj.get_logscale()
        self.remove_first_bin = obj.get_remove_first_bin()

    def update_hist(self, hist):
        hist.set_bins(self.n_bins)
        hist.set_logscale(self.logscale)
        hist.set_remove_first_bin(self.remove_first_bin)


# ===================================================
# Histogram 2D parameters
# ===================================================
class Histogram2DParam(ImageParam):
    """Histogram"""
    _multiselection = False
    _hide_background = True
    label = StringItem(_("Title"), default=_("Histogram")) \
            .set_prop("display", hide=GetAttrProp("_multiselection"))
    nx_bins = IntItem(_("X-axis bins"), default=100, min=1,
                      help=_("Number of bins along x-axis"))
    ny_bins = IntItem(_("Y-axis bins"), default=100, min=1,
                      help=_("Number of bins along y-axis"))
    logscale = BoolItem(_("logarithmic"), _("Z-axis scale"), default=False)
    
    def update_param(self, obj):
        super(Histogram2DParam, self).update_param(obj)
        self.logscale = obj.logscale
        self.nx_bins, self.ny_bins = obj.nx_bins, obj.ny_bins

    def update_histogram(self, histogram):
        histogram.logscale = int(self.logscale)
        histogram.set_bins(self.nx_bins, self.ny_bins)
        self.update_image(histogram)

class Histogram2DParam_MS(Histogram2DParam):
    _multiselection = True

ItemParameters.register_multiselection(Histogram2DParam, Histogram2DParam_MS)
    

# ===================================================
# Common shape parameters
# ===================================================
class ShapeParam(DataSet):
    _styles = BeginTabGroup("Styles")
    #------------------------------------------------------------------ Line tab
    ___line = BeginGroup(_("Line")).set_prop("display", icon="dashdot.png")
    line = LineStyleItem(_("Line (not selected)"))
    sel_line = LineStyleItem(_("Line (selected)"))
    ___eline = EndGroup(_("Line"))
    #---------------------------------------------------------------- Symbol tab
    ___sym = BeginGroup(_("Symbol")).set_prop("display", icon="diamond.png")
    symbol = SymbolItem(_("Symbol (not selected)"))
    sel_symbol = SymbolItem(_("Symbol (selected)"))
    ___esym = EndGroup(_("Symbol"))
    #------------------------------------------------------------------ Fill tab
    ___fill = BeginGroup(_("Fill pattern")).set_prop("display",
                                                     icon="dense6pattern.png")
    fill = BrushStyleItem(_("Fill pattern (not selected)"))
    sel_fill = BrushStyleItem(_("Fill pattern (selected)"))
    ___efill = EndGroup(_("Fill pattern"))
    #----------------------------------------------------------------------- End
    _endstyles = EndTabGroup("Styles")
    
    def update_param(self, obj):
        self.line.update_param(obj.pen)
        self.symbol.update_param(obj.symbol)
        self.fill.update_param(obj.brush)
        self.sel_line.update_param(obj.sel_pen)
        self.sel_symbol.update_param(obj.sel_symbol)
        self.sel_fill.update_param(obj.sel_brush)
        
    def update_shape(self, obj):
        obj.pen = self.line.build_pen()
        obj.symbol = self.symbol.build_symbol()
        obj.brush = self.fill.build_brush()
        obj.sel_pen = self.sel_line.build_pen()
        obj.sel_symbol = self.sel_symbol.build_symbol()
        obj.sel_brush = self.sel_fill.build_brush()

class AxesShapeParam(DataSet):
    arrow_angle = FloatItem(_("Arrow angle (°)"), min=0, max=90, nonzero=True)
    arrow_size = FloatItem(_("Arrow size (%)"), min=0, max=100, nonzero=True)
    _styles = BeginTabGroup("Styles")
    #------------------------------------------------------------------ Line tab
    ___line = BeginGroup(_("Line")).set_prop("display", icon="dashdot.png")
    xarrow_pen = LineStyleItem(_("Line (X-Axis)"))
    yarrow_pen = LineStyleItem(_("Line (Y-Axis)"))
    ___eline = EndGroup(_("Line"))
    #------------------------------------------------------------------ Fill tab
    ___fill = BeginGroup(_("Fill pattern")).set_prop("display",
                                                     icon="dense6pattern.png")
    xarrow_brush = BrushStyleItem(_("Fill pattern (X-Axis)"))
    yarrow_brush = BrushStyleItem(_("Fill pattern (Y-Axis)"))
    ___efill = EndGroup(_("Fill pattern"))
    #----------------------------------------------------------------------- End
    _endstyles = EndTabGroup("Styles")
    
    def update_param(self, obj):
        self.arrow_angle = obj.arrow_angle
        self.arrow_size = obj.arrow_size
        self.xarrow_pen.update_param(obj.x_pen)
        self.yarrow_pen.update_param(obj.y_pen)
        self.xarrow_brush.update_param(obj.x_brush)
        self.yarrow_brush.update_param(obj.y_brush)
        
    def update_axes(self, obj):
        obj.arrow_angle = self.arrow_angle
        obj.arrow_size = self.arrow_size
        obj.x_pen = self.xarrow_pen.build_pen()
        obj.x_brush = self.xarrow_brush.build_brush()
        obj.y_pen = self.yarrow_pen.build_pen()
        obj.y_brush = self.yarrow_brush.build_brush()

class AnnotationParam(DataSet):
    show_label = BoolItem(_("Show annotation"), default=True)
    show_position = BoolItem(_("Show position and size"), default=True)
    title = StringItem(_("Title"), default=u"")
    subtitle = StringItem(_("Subtitle"), default=u"")
    format = StringItem(_("String formatting"), default="%d pixels")
    transform_matrix = FloatArrayItem(_("Transform matrix"),
                                      default=np.eye(3, dtype=float))
    
    def update_param(self, obj):
        self.show_label = obj.is_label_visible()
        self.show_position = obj.position_and_size_visible
        self.title = obj.title().text()
        
    def update_annotation(self, obj):
        obj.setTitle(self.title)
        obj.set_label_visible(self.show_label)
        obj.position_and_size_visible = self.show_position
        obj.update_label()


# ===================================================
# Range selection parameters
# ===================================================
class RangeShapeParam(DataSet):
    _styles = BeginTabGroup("Styles")
    #------------------------------------------------------------------ Line tab
    ___line = BeginGroup(_("Line")).set_prop("display", icon="dashdot.png")
    line = LineStyleItem(_("Line (not selected)"))
    sel_line = LineStyleItem(_("Line (selected)"))
    ___eline = EndGroup(_("Line"))
    #---------------------------------------------------------------- Symbol tab
    ___symbol = BeginGroup(_("Symbol")).set_prop("display", icon="diamond.png")
    symbol = SymbolItem(_("Symbol (not selected)"))
    sel_symbol = SymbolItem(_("Symbol (selected)"))
    ___esymbol = EndGroup(_("Symbol"))
    #------------------------------------------------------------------ Fill tab
    ___fill = BeginGroup(_("Fill")).set_prop("display",
                                             icon="dense6pattern.png")
    fill = ColorItem(_("Fill color"))
    shade = FloatItem(_("Shade"), default = .05, min=0, max=1)
    ___efill = EndGroup(_("Fill"))
    #----------------------------------------------------------------------- End
    _endstyles = EndTabGroup("Styles")
    
    def update_param(self, range):
        self.line.update_param(range.pen)
        self.sel_line.update_param(range.sel_pen)
        self.fill = range.brush.color().name()
        self.shade = range.brush.color().alphaF()
        self.symbol.update_param(range.symbol)
        self.sel_symbol.update_param(range.sel_symbol)
        
    def update_range(self, range):
        range.pen = self.line.build_pen()
        range.sel_pen = self.sel_line.build_pen()
        col = QColor(self.fill)
        col.setAlphaF(self.shade)
        range.brush = QBrush(col)
        range.symbol = self.symbol.build_symbol()
        range.sel_symbol = self.sel_symbol.build_symbol()

