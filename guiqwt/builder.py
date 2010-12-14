# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut <pierre.raybaut@cea.fr>
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.builder
--------------

The `builder` module provides a builder singleton class 
used to simplify the creation of plot items.

Example
~~~~~~~

Before creating any widget, a `QApplication` must be instantiated 
(that is a `Qt` internal requirement):
          
>>> import guidata
>>> app = guidata.qapplication()

that is mostly equivalent to the following (the only difference is that 
the `guidata` helper function also installs the `Qt` translation 
corresponding to the system locale):
          
>>> from PyQt4.QtGui import QApplication
>>> app = QApplication([])

now that a `QApplication` object exists, we may create the plotting widget:

>>> from guiqwt.plot import ImageWidget
>>> widget = ImageWidget()

create curves, images, histograms, etc. and attach them to the plot:

>>> from guiqwt.builder import make
>>> curve = make.mcure(x, y, 'r+')
>>> image = make.image(data)
>>> hist = make.histogram(data, 100)
>>> for item in (curve, image, hist):
...     widget.plot.add_item()

and then show the widget to screen:

>>> widget.show()
>>> app.exec_()

Reference
~~~~~~~~~

.. autoclass:: PlotItemBuilder
   :members:
"""

from numpy import arange, array, zeros, meshgrid, ndarray

from PyQt4.Qwt5 import QwtPlot

# Local imports
from guiqwt.config import _, CONF, make_title
from guiqwt.curve import CurveItem, ErrorBarCurveItem, GridItem
from guiqwt.histogram import HistogramItem
from guiqwt.image import (ImageItem, QuadGridItem, TrImageItem, XYImageItem,
                          Histogram2DItem)
from guiqwt.shapes import (XRangeSelection, RectangleShape, EllipseShape,
                           SegmentShape)
from guiqwt.annotations import (AnnotatedRectangle, AnnotatedEllipse,
                                AnnotatedSegment)
from guiqwt.styles import (update_style_attr, CurveParam, ErrorBarParam,
                           style_generator, LabelParam, LegendParam, ImageParam,
                           TrImageParam, HistogramParam, Histogram2DParam,
                           ImageFilterParam, MARKERS, COLORS, GridParam,
                           LineStyleParam, AnnotationParam,
                           LabelParamWithContents)
from guiqwt.label import (LabelItem, LegendBoxItem, RangeComputation,
                          RangeComputation2d, DataInfoLabel,
                          SelectedLegendBoxItem)
from guiqwt.io import imagefile_to_array
import os.path as osp

# default offset positions for anchors
ANCHOR_OFFSETS = {
                  "TL" : ( 5,  5),
                  "TR" : (-5,  5),
                  "BL" : ( 5, -5),
                  "BR" : (-5, -5),
                  "L"  : ( 5,  0),
                  "R"  : (-5,  0),
                  "T"  : ( 0,  5),
                  "B"  : ( 0, -5),
                  }

CURVE_COUNT = 0
HISTOGRAM_COUNT = 0
IMAGE_COUNT = 0
LABEL_COUNT = 0
HISTOGRAM2D_COUNT = 0

class PlotItemBuilder(object):
    """
    This is just a bare class used to regroup
    a set of factory functions in a single object
    """
    AXES = {
            'bottom': QwtPlot.xBottom,
            'left'  : QwtPlot.yLeft,
            'top'   : QwtPlot.xTop,
            'right' : QwtPlot.yRight,
            }
    
    def __init__(self):
        self.style = style_generator()
        
    def gridparam(self, background=None,
                  major_enabled=None, minor_enabled=None,
                  major_style=None, minor_style=None):
        """
        Make `guiqwt.styles.GridParam` instance
           * background = canvas background color
           * major_enabled = tuple (major_xenabled, major_yenabled)
           * minor_enabled = tuple (minor_xenabled, minor_yenabled)
           * major_style = tuple (major_xstyle, major_ystyle)
           * minor_style = tuple (minor_xstyle, minor_ystyle)
           
        Style: tuple (style, color, width)
        """
        gridparam = GridParam(title=_("Grid"), icon="lin_lin.png")
        gridparam.read_config(CONF, "plot", "grid")
        if background is not None:
            gridparam.background = background
        if major_enabled is not None:
            gridparam.maj_xenabled, gridparam.maj_yenabled = major_enabled
        if minor_enabled is not None:
            gridparam.min_xenabled, gridparam.min_yenabled = minor_enabled
        if major_style is not None:
            style = LineStyleParam()
            linestyle, color, style.width = major_style
            style.set_style_from_matlab(linestyle)
            style.color = COLORS.get(color, color) # MATLAB-style
        if minor_style is not None:
            style = LineStyleParam()
            linestyle, color, style.width = minor_style
            style.set_style_from_matlab(linestyle)
            style.color = COLORS.get(color, color) # MATLAB-style
        return gridparam
    
    def grid(self, background=None, major_enabled=None, minor_enabled=None,
             major_style=None, minor_style=None):
        """
        Make a grid `plot item` (`guiqwt.curve.GridItem` object)
           * background = canvas background color
           * major_enabled = tuple (major_xenabled, major_yenabled)
           * minor_enabled = tuple (minor_xenabled, minor_yenabled)
           * major_style = tuple (major_xstyle, major_ystyle)
           * minor_style = tuple (minor_xstyle, minor_ystyle)
           
        Style: tuple (style, color, width)
        """
        gridparam = self.gridparam(background, major_enabled, minor_enabled,
                                   major_style, minor_style)
        return GridItem(gridparam)
    
    def __set_axes(self, curve, xaxis, yaxis):
        """Set curve axes"""
        for axis in (xaxis, yaxis):
            if axis not in self.AXES:
                raise RuntimeError("Unknown axis %s" % axis)
        curve.setXAxis(self.AXES[xaxis])
        curve.setYAxis(self.AXES[yaxis])

    def __set_param(self, param, title, color, linestyle, linewidth,
                    marker, markersize, markerfacecolor, markeredgecolor,
                    shade, fitted, curvestyle, curvetype, baseline):
        """Apply parameters to a `guiqwt.styles.CurveParam` instance"""
        if title:
            param.label = title
        if color is not None:
            color = COLORS.get(color, color) # MATLAB-style
            param.line.color = color
        if linestyle is not None:
            param.line.set_style_from_matlab(linestyle)
        if linewidth is not None:
            param.line.width = linewidth
        if marker is not None:
            if marker in MARKERS:
                param.symbol.update_param(MARKERS[marker]) # MATLAB-style
            else:
                param.symbol.marker = marker
        if markersize is not None:
            param.symbol.size = markersize
        if markerfacecolor is not None:
            markerfacecolor = COLORS.get(markerfacecolor,
                                         markerfacecolor) # MATLAB-style
            param.symbol.facecolor = markerfacecolor
        if markeredgecolor is not None:
            markeredgecolor = COLORS.get(markeredgecolor,
                                         markeredgecolor) # MATLAB-style
            param.symbol.edgecolor = markeredgecolor
        if shade is not None:
            param.shade = shade
        if fitted is not None:
            param.fitted = fitted
        if curvestyle is not None:
            param.curvestyle = curvestyle
        if curvetype is not None:
            param.curvetype = curvetype
        if baseline is not None:
            param.baseline = baseline
            
    def __get_arg_triple_plot(self, args):
        """Convert MATLAB-like arguments into x, y, style"""
        def get_x_y_from_data(data):
            if len(data.shape) == 1 or data.shape[0] == 1 or data.shape[1] == 1:
                x = arange(data.size)
                y = data
            else:
                x = arange(len(data[:, 0]))
                y = [data[:, i] for i in range(len(data[0, :]))]
            return x, y
            
        if len(args)==1:
            if isinstance(args[0], basestring):
                x = array((), float)
                y = array((), float)
                style = args[0]
            else:
                x, y = get_x_y_from_data(args[0])
                if isinstance(y, ndarray):
                    style = self.style.next()
                else:
                    style = [self.style.next() for yi in y]
        elif len(args)==2:
            a1, a2 = args
            if isinstance(a2, basestring):
                x, y = get_x_y_from_data(a1)
                style = a2
            else:
                x = a1
                y = a2
                style = self.style.next()
        elif len(args)==3:
            x, y, style = args
        else:
            raise TypeError("Wrong number of arguments")
        return x, y, style
        
    def __get_arg_triple_errorbar(self, args):
        """Convert MATLAB-like arguments into x, y, style"""
        if len(args)==2:
            y, dy = args
            x = arange(len(y))
            dx = zeros(len(y))
            style = self.style.next()
        elif len(args)==3:
            a1, a2, a3 = args
            if isinstance(a3, basestring):
                y, dy = a1, a2
                x = arange(len(y))
                dx = zeros(len(y))
                style = a3
            else:
                x, y, dy = args
                dx = zeros(len(y))
                style = self.style.next()
        elif len(args)==4:
            a1, a2, a3, a4 = args
            if isinstance(a4, basestring):
                x, y, dy = a1, a2, a3
                dx = zeros(len(y))
                style = a4
            else:
                x, y, dx, dy = args
                style = self.style.next()
        elif len(args)==5:
            x, y, dx, dy, style = args
        else:
            raise TypeError("Wrong number of arguments")
        return x, y, dx, dy, style
    
    def mcurve(self, *args, **kwargs):
        """
        Make a curve `plot item` based on MATLAB-like syntax
        (may returns a list of curves if data contains more than one signal)
        (:py:class:`guiqwt.curve.CurveItem` object)
        
        Example: mcurve(x, y, 'r+')
        """
        x, y, style = self.__get_arg_triple_plot(args)
        if isinstance(y, ndarray):
            y = [y]
        if not isinstance(style, list):
            style = [style]
        if len(y) > len(style):
            style = [style[0]]*len(y)
        basename = _("Curve")
        curves = []
        for yi, stylei in zip(y, style):
            param = CurveParam(title=basename, icon='curve.png')
            if "label" in kwargs:
                param.label = kwargs.pop("label")
            else:
                global CURVE_COUNT
                CURVE_COUNT += 1
                param.label = make_title(basename, CURVE_COUNT)
            update_style_attr(stylei, param)
            curves.append(self.pcurve(x, yi, param, **kwargs))
        if len(curves) == 1:
            return curves[0]
        else:
            return curves
                
    def pcurve(self, x, y, param, xaxis="bottom", yaxis="left"):
        """
        Make a curve `plot item` 
        based on a `guiqwt.styles.CurveParam` instance
        (:py:class:`guiqwt.curve.CurveItem` object)
        
        Usage: pcurve(x, y, param)
        """
        curve = CurveItem(param)
        curve.set_data(x, y)
        curve.update_params()
        self.__set_axes(curve, xaxis, yaxis)
        return curve

    def curve(self, x, y, title=u"",
              color=None, linestyle=None, linewidth=None,
              marker=None, markersize=None, markerfacecolor=None,
              markeredgecolor=None, shade=None, fitted=None,
              curvestyle=None, curvetype=None, baseline=None,
              xaxis="bottom", yaxis="left"):
        """
        Make a curve `plot item` from x, y, data
        (:py:class:`guiqwt.curve.CurveItem` object)
            * x: 1D NumPy array
            * y: 1D NumPy array
            * color: curve color name
            * linestyle: curve line style (MATLAB-like string or attribute name 
              from the :py:class:`PyQt4.QtCore.Qt.PenStyle` enum
              (i.e. "SolidLine" "DashLine", "DotLine", "DashDotLine", 
              "DashDotDotLine" or "NoPen")
            * linewidth: line width (pixels)
            * marker: marker shape (MATLAB-like string or attribute name from 
              the :py:class:`PyQt4.Qwt5.QwtSymbol.Style` enum (i.e. "Cross",
              "Ellipse", "Star1", "XCross", "Rect", "Diamond", "UTriangle", 
              "DTriangle", "RTriangle", "LTriangle", "Star2" or "NoSymbol")
            * markersize: marker size (pixels)
            * markerfacecolor: marker face color name
            * markeredgecolor: marker edge color name
            * shade: 0 <= float <= 1 (curve shade)
            * fitted: boolean (fit curve to data)
            * curvestyle: attribute name from the 
              :py:class:`PyQt4.Qwt5.QwtPlotCurve.CurveStyle` enum
              (i.e. "Lines", "Sticks", "Steps", "Dots" or "NoCurve")
            * curvetype: attribute name from the 
              :py:class:`PyQt4.Qwt5.QwtPlotCurve.CurveType` enum
              (i.e. "Yfx" or "Xfy")
            * baseline (float: default=0.0): the baseline is needed for filling 
              the curve with a brush or the Sticks drawing style. 
              The interpretation of the baseline depends on the curve type 
              (horizontal line for "Yfx", vertical line for "Xfy")
            * xaxis, yaxis: X/Y axes bound to curve
        
        Examples:
        curve(x, y, marker='Ellipse', markerfacecolor='#ffffff')
        which is equivalent to (MATLAB-style support):
        curve(x, y, marker='o', markerfacecolor='w')
        """
        basename = _("Curve")
        param = CurveParam(title=basename, icon='curve.png')
        if not title:
            global CURVE_COUNT
            CURVE_COUNT += 1
            title = make_title(basename, CURVE_COUNT)
        self.__set_param(param, title, color, linestyle, linewidth, marker,
                         markersize, markerfacecolor, markeredgecolor,
                         shade, fitted, curvestyle, curvetype, baseline)
        return self.pcurve(x, y, param, xaxis, yaxis)

    def merror(self, *args, **kwargs):
        """
        Make an errorbar curve `plot item` based on MATLAB-like syntax
        (:py:class:`guiqwt.curve.ErrorBarCurveItem` object)
        
        Example: mcurve(x, y, 'r+')
        """
        x, y, dx, dy, style = self.__get_arg_triple_errorbar(args)
        basename = _("Curve")
        curveparam = CurveParam(title=basename, icon='curve.png')
        errorbarparam = ErrorBarParam(title=_("Error bars"),
                                      icon='errorbar.png')
        if "label" in kwargs:
            curveparam.label = kwargs["label"]
        else:
            global CURVE_COUNT
            CURVE_COUNT += 1
            curveparam.label = make_title(basename, CURVE_COUNT)
        update_style_attr(style, curveparam)
        errorbarparam.color = curveparam.line.color
        return self.perror(x, y, dx, dy, curveparam, errorbarparam)

    def perror(self, x, y, dx, dy, curveparam, errorbarparam,
               xaxis="bottom", yaxis="left"):
        """
        Make an errorbar curve `plot item` 
        based on a `guiqwt.styles.ErrorBarParam` instance
        (:py:class:`guiqwt.curve.ErrorBarCurveItem` object)
            * x: 1D NumPy array
            * y: 1D NumPy array
            * dx: None, or scalar, or 1D NumPy array
            * dy: None, or scalar, or 1D NumPy array
            * curveparam: `guiqwt.styles.CurveParam` object
            * errorbarparam: `guiqwt.styles.ErrorBarParam` object
            * xaxis, yaxis: X/Y axes bound to curve
        
        Usage: perror(x, y, dx, dy, curveparam, errorbarparam)
        """
        curve = ErrorBarCurveItem(curveparam, errorbarparam)
        curve.set_data(x, y, dx, dy)
        curve.update_params()
        self.__set_axes(curve, xaxis, yaxis)
        return curve
        
    def error(self, x, y, dx, dy, title=u"",
              color=None, linestyle=None, linewidth=None, marker=None,
              markersize=None, markerfacecolor=None, markeredgecolor=None,
              shade=None, fitted=None, curvestyle=None, curvetype=None,
              baseline=None, xaxis="bottom", yaxis="left"):
        """
        Make an errorbar curve `plot item` 
        (:py:class:`guiqwt.curve.ErrorBarCurveItem` object)
            * x: 1D NumPy array
            * y: 1D NumPy array
            * dx: None, or scalar, or 1D NumPy array
            * dy: None, or scalar, or 1D NumPy array
            * color: curve color name
            * linestyle: curve line style (MATLAB-like string or attribute name 
              from the :py:class:`PyQt4.QtCore.Qt.PenStyle` enum
              (i.e. "SolidLine" "DashLine", "DotLine", "DashDotLine", 
              "DashDotDotLine" or "NoPen")
            * linewidth: line width (pixels)
            * marker: marker shape (MATLAB-like string or attribute name from 
              the :py:class:`PyQt4.Qwt5.QwtSymbol.Style` enum (i.e. "Cross",
              "Ellipse", "Star1", "XCross", "Rect", "Diamond", "UTriangle", 
              "DTriangle", "RTriangle", "LTriangle", "Star2" or "NoSymbol")
            * markersize: marker size (pixels)
            * markerfacecolor: marker face color name
            * markeredgecolor: marker edge color name
            * shade: 0 <= float <= 1 (curve shade)
            * fitted: boolean (fit curve to data)
            * curvestyle: attribute name from the 
              :py:class:`PyQt4.Qwt5.QwtPlotCurve.CurveStyle` enum
              (i.e. "Lines", "Sticks", "Steps", "Dots" or "NoCurve")
            * curvetype: attribute name from the 
              :py:class:`PyQt4.Qwt5.QwtPlotCurve.CurveType` enum
              (i.e. "Yfx" or "Xfy")
            * baseline (float: default=0.0): the baseline is needed for filling 
              the curve with a brush or the Sticks drawing style. 
              The interpretation of the baseline depends on the curve type 
              (horizontal line for "Yfx", vertical line for "Xfy")
            * xaxis, yaxis: X/Y axes bound to curve
        
        Examples::
            error(x, y, None, dy, marker='Ellipse', markerfacecolor='#ffffff')
            which is equivalent to (MATLAB-style support):
            error(x, y, None, dy, marker='o', markerfacecolor='w')
        """
        basename = _("Curve")
        curveparam = CurveParam(title=basename, icon='curve.png')
        errorbarparam = ErrorBarParam(title=_("Error bars"),
                                      icon='errorbar.png')
        if not title:
            global CURVE_COUNT
            CURVE_COUNT += 1
            curveparam.label = make_title(basename, CURVE_COUNT)
        self.__set_param(curveparam, title, color, linestyle, linewidth, marker,
                         markersize, markerfacecolor, markeredgecolor,
                         shade, fitted, curvestyle, curvetype, baseline)
        errorbarparam.color = curveparam.line.color
        return self.perror(x, y, dx, dy, curveparam, errorbarparam,
                           xaxis, yaxis)
    
    def histogram(self, data, bins=None, logscale=None, remove_first_bin=None,
                  title=u"", color=None, xaxis="bottom", yaxis="left"):
        """
        Make 1D Histogram `plot item` 
        (:py:class:`guiqwt.histogram.HistogramItem` object)
            * data (1D NumPy array)
            * bins: number of bins (int)
            * logscale: Y-axis scale (bool)
        """
        basename = _("Histogram")
        histparam = HistogramParam(title=basename, icon='histogram.png')
        curveparam = CurveParam(_("Curve"), icon='curve.png')
        curveparam.read_config(CONF, "histogram", "curve")
        if not title:
            global HISTOGRAM_COUNT
            HISTOGRAM_COUNT += 1
            title = make_title(basename, HISTOGRAM_COUNT)
        curveparam.label = title
        if color is not None:
            curveparam.line.color = color
        if bins is not None:
            histparam.n_bins = bins
        if logscale is not None:
            histparam.logscale = logscale
        if remove_first_bin is not None:
            histparam.remove_first_bin = remove_first_bin
        return self.phistogram(data, curveparam, histparam, xaxis, yaxis)
        
    def phistogram(self, data, curveparam, histparam,
                   xaxis="bottom", yaxis="left"):
        """
        Make 1D histogram `plot item` 
        (:py:class:`guiqwt.histogram.HistogramItem` object) 
        based on a `guiqwt.styles.CurveParam` and 
        `guiqwt.styles.HistogramParam` instances
        
        Usage: phistogram(data, curveparam, histparam)
        """
        hist = HistogramItem(curveparam, histparam)
        hist.update_params()
        hist.set_hist_data(data)
        self.__set_axes(hist, xaxis, yaxis)
        return hist

    def __set_image_param(self, param, title, background_color,
                          alpha_mask, alpha, colormap, **kwargs):
        if title:
            param.label = title
        else:
            global IMAGE_COUNT
            IMAGE_COUNT += 1
            param.label = make_title(_("Image"), IMAGE_COUNT)
        if background_color is not None:
            param.background = background_color
        if alpha_mask is not None:
            param.alpha_mask = alpha_mask
        if alpha is not None:
            param.alpha = alpha
        if colormap is not None:
            param.colormap = colormap
        for key, val in kwargs.items():
            setattr(param, key, val)

    def _get_image_data(self, data, filename, title, cmap):
        if data is None:
            assert filename is not None
            data = imagefile_to_array(filename)
        if title is None and filename is not None:
            title = osp.basename(filename)
        return data, filename, title, cmap

    def image(self, data=None, filename=None, title=None, background_color=None,
              alpha_mask=None, alpha=None, colormap=None,
              xaxis="bottom", yaxis="left", zaxis="right"):
        """
        Make an image `plot item` from data
        (:py:class:`guiqwt.image.ImageItem` object)
        """
        param = ImageParam(title=_("Image"), icon='image.png')
        params = self._get_image_data(data, filename, title, colormap)
        data, filename, title, colormap = params
        self.__set_image_param(param, title, background_color,
                               alpha_mask, alpha, colormap)
        image = ImageItem(data, param)
        image.set_filename(filename)
        return image
        
    def quadgrid(self, X, Y, Z, filename=None, title=None,
                 background_color=None, alpha_mask=None, alpha=None,
                 colormap=None, xaxis="bottom", yaxis="left", zaxis="right"):
        """
        Make a pseudocolor `plot item` of a 2D array
        (:py:class:`guiqwt.image.QuadGridItem` object)
        """
        param = ImageParam(title=_("Image"), icon='image.png')
        self.__set_image_param(param, title, background_color,
                               alpha_mask, alpha, colormap)
        image = QuadGridItem(X, Y, Z, param)
        return image

    def pcolor(self, *args, **kwargs):
        """
        Make a pseudocolor `plot item` of a 2D array 
        based on MATLAB-like syntax
        (:py:class:`guiqwt.image.QuadGridItem` object)
        
        Examples:
            pcolor(C)
            pcolor(X, Y, C)
        """
        if len(args) == 1:
            Z, = args
            M, N = Z.shape
            X, Y = meshgrid(arange(N, dtype=Z.dtype), arange(M, dtype=Z.dtype))
        elif len(args) == 3:
            X, Y, Z = args
        else:
            raise RuntimeError("1 or 3 non-keyword arguments expected")
        return self.quadgrid(X, Y, Z, **kwargs)

    def trimage(self, data=None, filename=None, title=None,
                background_color=None, alpha_mask=None, alpha=None,
                colormap=None, xaxis="bottom", yaxis="left", zaxis="right",
                x0=0.0, y0=0.0, angle=0.0, dx=1.0, dy=1.0,
                interpolation='linear'):
        """
        Make a transformable image `plot item` (image with an arbitrary 
        affine transform)
        (:py:class:`guiqwt.image.TrImageItem` object)
            * data: 2D NumPy array (image pixel data)
            * filename: image filename (if data is not specified)
            * title: image title (optional)
            * x0, y0: position
            * angle: angle (radians)
            * dx, dy: pixel size along X and Y axes
            * interpolation: 'nearest', 'linear' (default), 'antialiasing' (5x5)
        """
        param = TrImageParam(title=_("Image"), icon='image.png')
        params = self._get_image_data(data, filename, title, colormap)
        data, filename, title, colormap = params
        self.__set_image_param(param, title, background_color,
                               alpha_mask, alpha, colormap,
                               x0=x0, y0=y0, angle=angle, dx=dx, dy=dy)
        interp_methods = {'nearest': 0, 'linear': 1, 'antialiasing': 5}
        param.interpolation = interp_methods[interpolation]
        image = TrImageItem(data, param)
        image.set_filename(filename)
        return image

    def xyimage(self, x, y, data, title=None, background_color=None,
                alpha_mask=None, alpha=None, colormap=None,
                xaxis="bottom", yaxis="left", zaxis="right"):
        """
        Make an xyimage `plot item` (image with non-linear X/Y axes) from data
        (:py:class:`guiqwt.image.XYImageItem` object)
            * x: 1D NumPy array
            * y: 1D NumPy array
            * data: 2D NumPy array (image pixel data)
            * title: image title (optional)
        """
        param = ImageParam(title=_("Image"), icon='image.png')
        self.__set_image_param(param, title, background_color,
                               alpha_mask, alpha, colormap)
        return XYImageItem(x, y, data, param)
    
    def imagefilter(self, xmin, xmax, ymin, ymax,
                    imageitem, filter, title=None):
        """
        Make a rectangular area image filter `plot item`
        (:py:class:`guiqwt.image.ImageFilterItem` object)
            * xmin, xmax, ymin, ymax: filter area bounds
            * imageitem: An imageitem instance
            * filter: function (x, y, data) --> data
        """
        param = ImageFilterParam(_("Filter"), icon="funct.png")
        param.xmin, param.xmax, param.ymin, param.ymax = xmin, xmax, ymin, ymax
        if title is not None:
            param.label = title
        filt = imageitem.get_filter(filter, param)
        _m, _M = imageitem.get_lut_range()
        filt.set_lut_range([_m, _M])
        return filt
    
    def histogram2D(self, X, Y, NX=None, NY=None, logscale=None,
                    title=None, transparent=None):
        """
        Make a 2D Histogram `plot item` 
        (:py:class:`guiqwt.image.Histogram2DItem` object)
            * X: data (1D array)
            * Y: data (1D array)
            * NX: Number of bins along x-axis (int)
            * NY: Number of bins along y-axis (int)
            * logscale: Z-axis scale (bool)
            * title: item title (string)
            * transparent: enable transparency (bool)
        """
        basename = _("2D Histogram")
        param = Histogram2DParam(title=basename, icon='histogram2d.png')
        if NX is not None:
            param.nx_bins = NX
        if NY is not None:
            param.ny_bins = NY
        if logscale is not None:
            param.logscale = int(logscale)
        if title is not None:
            param.label = title
        else:
            global HISTOGRAM2D_COUNT
            HISTOGRAM2D_COUNT += 1
            param.label = make_title(basename, HISTOGRAM2D_COUNT)
        if transparent is not None:
            param.transparent = transparent
        return Histogram2DItem(X, Y, param)

    def label(self, text, g, c, anchor, title=""):
        """
        Make a label `plot item` 
        (:py:class:`guiqwt.label.LabelItem` object)
            * text: label text (string)
            * g: position in plot coordinates (tuple) 
              or relative position (string)
            * c: position in canvas coordinates (tuple)
            * anchor: anchor position in relative position (string)
            * title: label name (optional)
        
        Examples::
            make.label("Relative position", (x[0], y[0]), (10, 10), "BR")
            make.label("Absolute position", "R", (0,0), "R")
        """
        basename = _("Label")
        param = LabelParamWithContents(basename, icon='label.png')
        param.read_config(CONF, "plot", "label")
        if title:
            param.label = title
        else:
            global LABEL_COUNT
            LABEL_COUNT += 1
            param.label = make_title(basename, LABEL_COUNT)
        if isinstance(g, tuple):
            param.abspos = False
            param.xg, param.yg = g
        else:
            param.abspos = True
            param.absg = g
        if c is None:
            c = ANCHOR_OFFSETS[anchor]
        param.xc, param.yc = c
        param.anchor = anchor
        return LabelItem(text, param)

    def legend(self, anchor='TR', c=None, restrict_items=None):
        """
        Make a legend `plot item` 
        (:py:class:`guiqwt.label.LegendBoxItem` or 
        :py:class:`guiqwt.label.SelectedLegendBoxItem` object)
            * anchor: legend position in relative position (string)
            * c (optional): position in canvas coordinates (tuple)
            * restrict_items (optional):
                - None: all items are shown in legend box
                - []: no item shown
                - [item1, item2]: item1, item2 are shown in legend box
        """
        param = LegendParam(_("Legend"), icon='legend.png')
        param.read_config(CONF, "plot", "legend")
        param.abspos = True
        param.absg = anchor
        param.anchor = anchor
        if c is None:
            c = ANCHOR_OFFSETS[anchor]
        param.xc, param.yc = c
        if restrict_items is None:
            return LegendBoxItem(param)
        else:
            return SelectedLegendBoxItem(param, restrict_items)

    def range(self, xmin, xmax):
        return XRangeSelection(xmin, xmax)
        
    def __shape(self, shapeclass, x0, y0, x1, y1, title=None):
        shape = shapeclass(x0, y0, x1, y1)
        shape.set_style("plot", "shape/drag")
        if title is not None:
            shape.setTitle(title)
        return shape

    def rectangle(self, x0, y0, x1, y1, title=None):
        """
        Make a rectangle shape `plot item` 
        (:py:class:`guiqwt.shapes.RectangleShape` object)
            * x0, y0, x1, y1: rectangle coordinates
            * title: label name (optional)
        """
        return self.__shape(RectangleShape, x0, y0, x1, y1, title)

    def ellipse(self, x0, y0, x1, y1, ratio, title=None):
        """
        Make an ellipse shape `plot item` 
        (:py:class:`guiqwt.shapes.EllipseShape` object)
            * x0, y0, x1, y1: ellipse x-axis coordinates
            * ratio: ratio between y-axis and x-axis lengths
            * title: label name (optional)
        """
        shape = EllipseShape(x0, y0, x1, y1, ratio)
        shape.set_style("plot", "shape/drag")
        if title is not None:
            shape.setTitle(title)
        return shape
        
    def circle(self, x0, y0, x1, y1, title=None):
        """
        Make a circle shape `plot item` 
        (:py:class:`guiqwt.shapes.EllipseShape` object)
            * x0, y0, x1, y1: circle diameter coordinates
            * title: label name (optional)
        """
        return self.ellipse(x0, y0, x1, y1, 1., title=title)

    def segment(self, x0, y0, x1, y1, title=None):
        """
        Make a segment shape `plot item` 
        (:py:class:`guiqwt.shapes.SegmentShape` object)
            * x0, y0, x1, y1: segment coordinates
            * title: label name (optional)
        """
        return self.__shape(SegmentShape, x0, y0, x1, y1, title)
        
    def __get_annotationparam(self, title, subtitle):
        param = AnnotationParam(_("Annotation"), icon="annotation.png")
        if title is not None:
            param.title = title
        if subtitle is not None:
            param.subtitle = subtitle
        return param
        
    def __annotated_shape(self, shapeclass, x0, y0, x1, y1, title, subtitle):
        param = self.__get_annotationparam(title, subtitle)
        shape = shapeclass(x0, y0, x1, y1, param)
        shape.set_style("plot", "shape/drag")
        return shape
        
    def annotated_rectangle(self, x0, y0, x1, y1, title=None, subtitle=None):
        """
        Make an annotated rectangle `plot item` 
        (:py:class:`guiqwt.annotations.AnnotatedRectangle` object)
            * x0, y0, x1, y1: rectangle coordinates
            * title, subtitle: strings
        """
        return self.__annotated_shape(AnnotatedRectangle,
                                      x0, y0, x1, y1, title, subtitle)
        
    def annotated_ellipse(self, x0, y0, x1, y1, ratio,
                          title=None, subtitle=None):
        """
        Make an annotated ellipse `plot item`
        (:py:class:`guiqwt.annotations.AnnotatedEllipse` object)
            * x0, y0, x1, y1: ellipse rectangle coordinates
            * ratio: ratio between y-axis and x-axis lengths
            * title, subtitle: strings
        """
        param = self.__get_annotationparam(title, subtitle)
        shape = AnnotatedEllipse(x0, y0, x1, y1, ratio, param)
        shape.set_style("plot", "shape/drag")
        return shape
                                      
    def annotated_circle(self, x0, y0, x1, y1, ratio,
                         title=None, subtitle=None):
        """
        Make an annotated circle `plot item`
        (:py:class:`guiqwt.annotations.AnnotatedCircle` object)
            * x0, y0, x1, y1: circle diameter coordinates
            * title, subtitle: strings
        """
        return self.annotated_ellipse(x0, y0, x1, y1, 1., title, subtitle)
        
    def annotated_segment(self, x0, y0, x1, y1, title=None, subtitle=None):
        """
        Make an annotated segment `plot item`
        (:py:class:`guiqwt.annotations.AnnotatedSegment` object)
            * x0, y0, x1, y1: segment coordinates
            * title, subtitle: strings
        """
        return self.__annotated_shape(AnnotatedSegment,
                                      x0, y0, x1, y1, title, subtitle)

    def info_label(self, anchor, comps, title=""):
        """
        Make an info label `plot item` 
        (:py:class:`guiqwt.label.DataInfoLabel` object)
        """
        basename = _("Computation")
        param = LabelParam(basename, icon='label.png')
        param.read_config(CONF, "plot", "info_label")
        if title:
            param.label = title
        else:
            global LABEL_COUNT
            LABEL_COUNT += 1
            param.label = make_title(basename, LABEL_COUNT)
        param.abspos = True
        param.absg = anchor
        param.anchor = anchor
        c = ANCHOR_OFFSETS[anchor]
        param.xc, param.yc = c
        return DataInfoLabel(param, comps)

    def computation(self, range, anchor, label, curve, function):
        """
        Make a computation label `plot item` 
        (:py:class:`guiqwt.label.DataInfoLabel` object)
        (see example: :py:mod:`guiqwt.tests.computations`)
        """
        return self.computations(range, anchor, [ (curve, label, function) ])

    def computations(self, range, anchor, specs):
        """
        Make computation labels  `plot item` 
        (:py:class:`guiqwt.label.DataInfoLabel` object)
        (see example: :py:mod:`guiqwt.tests.computations`)
        """
        comps = []
        for curve, label, function in specs:
            comp = RangeComputation(label, curve, range, function)
            comps.append(comp)
        return self.info_label(anchor, comps)

    def computation2d(self, rect, anchor, label, image, function):
        """
        Make a 2D computation label `plot item` 
        (:py:class:`guiqwt.label.RangeComputation2d` object)
        (see example: :py:mod:`guiqwt.tests.computations`)
        """
        return self.computations2d(rect, anchor, [ (image, label, function) ])

    def computations2d(self, rect, anchor, specs):
        """
        Make 2D computation labels `plot item` 
        (:py:class:`guiqwt.label.RangeComputation2d` object)
        (see example: :py:mod:`guiqwt.tests.computations`)
        """
        comps = []
        for image, label, function in specs:
            comp = RangeComputation2d(label, image, rect, function)
            comps.append(comp)
        return self.info_label(anchor, comps)

make = PlotItemBuilder()