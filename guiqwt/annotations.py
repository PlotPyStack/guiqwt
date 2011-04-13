# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.annotations
------------------

The `annotations` module provides annotated shapes:
    * :py:class:`guiqwt.annotations.AnnotatedPoint`
    * :py:class:`guiqwt.annotations.AnnotatedSegment`
    * :py:class:`guiqwt.annotations.AnnotatedRectangle`
    * :py:class:`guiqwt.annotations.AnnotatedEllipse`
    * :py:class:`guiqwt.annotations.AnnotatedCircle`

An annotated shape is a plot item (derived from QwtPlotItem) that may be 
displayed on a 2D plotting widget like :py:class:`guiqwt.curve.CurvePlot` 
or :py:class:`guiqwt.image.ImagePlot`.

.. seealso:: module :py:mod:`guiqwt.shapes`

Examples
~~~~~~~~

An annotated shape may be created:
    * from the associated plot item class (e.g. `AnnotatedCircle` to 
      create an annotated circle): the item properties are then assigned 
      by creating the appropriate style parameters object
      (:py:class:`guiqwt.styles.AnnotationParam`)
      
>>> from guiqwt.annotations import AnnotatedCircle
>>> from guiqwt.styles import AnnotationParam
>>> param = AnnotationParam()
>>> param.title = 'My circle'
>>> circle_item = AnnotatedCircle(0., 2., 4., 0., param)
      
    * or using the `plot item builder` (see :py:func:`guiqwt.builder.make`):
      
>>> from guiqwt.builder import make
>>> circle_item = make.annotated_circle(0., 2., 4., 0., title='My circle')

Reference
~~~~~~~~~

.. autoclass:: AnnotatedPoint
   :members:
   :inherited-members:
.. autoclass:: AnnotatedSegment
   :members:
   :inherited-members:
.. autoclass:: AnnotatedRectangle
   :members:
   :inherited-members:
.. autoclass:: AnnotatedEllipse
   :members:
   :inherited-members:
.. autoclass:: AnnotatedCircle
   :members:
   :inherited-members:
"""

import numpy as np
from math import fabs

from guidata.utils import update_dataset

# Local imports
from guiqwt.config import CONF, _
from guiqwt.styles import LabelParam, AnnotationParam
from guiqwt.shapes import (AbstractShape, RectangleShape, EllipseShape,
                           SegmentShape, PointShape, VerticalCursor,
                           HorizontalCursor)
from guiqwt.label import DataInfoLabel
from guiqwt.interfaces import IShapeItemType, ISerializableType
from guiqwt.signals import (SIG_ANNOTATION_CHANGED, SIG_ITEM_MOVED,
                            SIG_CURSOR_MOVED)


class AnnotatedShape(AbstractShape):
    """
    Construct an annotated shape with properties set with
    *annotationparam* (see :py:class:`guiqwt.styles.AnnotationParam`)
    """
    SHAPE_CLASS = None
    LABEL_ANCHOR = None
    def __init__(self, annotationparam=None):
        AbstractShape.__init__(self)
        assert self.LABEL_ANCHOR is not None
        self.shape = self.create_shape()
        self.label = self.create_label()
        self.area_computations_visible = True
        if annotationparam is None:
            self.annotationparam = AnnotationParam(_("Annotation"),
                                                   icon="annotation.png")
        else:
            self.annotationparam = annotationparam
            self.annotationparam.update_annotation(self)
        
    def types(self):
        return (IShapeItemType, ISerializableType)
    
    def __reduce__(self):
        state = (self.shape, self.label, self.annotationparam)
        return (self.__class__, (), state)

    def __setstate__(self, state):
        shape, label, param = state
        self.shape = shape
        self.label = label
        self.annotationparam = param
        self.annotationparam.update_annotation(self)

    def set_style(self, section, option):
        self.shape.set_style(section, option)
        
    #----QwtPlotItem API--------------------------------------------------------
    def draw(self, painter, xMap, yMap, canvasRect):
        self.shape.draw(painter, xMap, yMap, canvasRect)
        if self.label.isVisible():
            self.label.draw(painter, xMap, yMap, canvasRect)
        
    #----Public API-------------------------------------------------------------
    def create_shape(self):
        """Return the shape object associated to this annotated shape object"""
        shape = self.SHAPE_CLASS(0, 0, 1, 1)
        return shape
        
    def create_label(self):
        """Return the label object associated to this annotated shape object"""
        label_param = LabelParam(_("Label"), icon='label.png')
        label_param.read_config(CONF, "plot", "shape/label")
        label_param.anchor = self.LABEL_ANCHOR
        return DataInfoLabel(label_param, [self])
        
    def is_label_visible(self):
        """Return True if associated label is visible"""
        return self.label.isVisible()
        
    def set_label_visible(self, state):
        """Set the annotated shape's label visibility"""
        self.label.setVisible(state)
        
    def update_label(self):
        """Update the annotated shape's label contents"""
        self.label.update_text()

    def get_text(self):
        """
        Return text associated to current shape
        (see :py:class:`guiqwt.label.ObjectInfo`)
        """
        text = ""
        title = self.title().text()
        if title:
            text += "<b>%s</b>" % title
        subtitle = self.annotationparam.subtitle
        if subtitle:
            if text:
                text += "<br>"
            text += "<i>%s</i>" % subtitle
        if self.area_computations_visible:
            infos = self.get_infos()
            if infos:
                if text:
                    text += "<br>"
                text += infos
        return text
        
    def get_infos(self):
        """Return formatted string with informations on current shape"""
        pass
        
    def set_label_position(self):
        """Set label position, for instance based on shape position"""
        raise NotImplementedError
    
    def apply_transform_matrix(self, x, y):
        V = np.array([x, y, 1.])
        W = np.dot(V, self.annotationparam.transform_matrix)
        return W[0], W[1]
    
    def get_transformed_coords(self, handle1, handle2):
        x1, y1 = self.apply_transform_matrix(*self.shape.points[handle1])
        x2, y2 = self.apply_transform_matrix(*self.shape.points[handle2])
        return x1, y1, x2, y2

    #----IBasePlotItem API------------------------------------------------------
    def hit_test(self, pos):
        return self.shape.poly_hit_test(self.plot(),
                                        self.xAxis(), self.yAxis(), pos)
            
    def move_point_to(self, handle, pos, ctrl=None):
        self.shape.move_point_to(handle, pos, ctrl)
        self.set_label_position()
        if self.plot():
            self.plot().emit(SIG_ANNOTATION_CHANGED, self)

    def move_shape(self, old_pos, new_pos):
        self.shape.move_shape(old_pos, new_pos)
        self.label.move_local_shape(old_pos, new_pos)
        
    def move_local_shape(self, old_pos, new_pos):
        old_pt = self.canvas_to_axes(old_pos)
        new_pt = self.canvas_to_axes(new_pos)
        self.shape.move_shape(old_pt, new_pt)
        self.set_label_position()
        if self.plot():
            self.plot().emit(SIG_ITEM_MOVED, self, *(old_pt+new_pt))
            self.plot().emit(SIG_ANNOTATION_CHANGED, self)

    def select(self):
        """Select item"""
        super(AnnotatedShape, self).select()
        self.shape.select()
    
    def unselect(self):
        """Unselect item"""
        super(AnnotatedShape, self).unselect()
        self.shape.unselect()

    def get_item_parameters(self, itemparams):
        self.shape.get_item_parameters(itemparams)
        self.label.get_item_parameters(itemparams)
        self.annotationparam.update_param(self)
        itemparams.add("AnnotationParam", self, self.annotationparam)
    
    def set_item_parameters(self, itemparams):
        self.shape.set_item_parameters(itemparams)
        self.label.set_item_parameters(itemparams)
        update_dataset(self.annotationparam, itemparams.get("AnnotationParam"),
                       visible_only=True)
        self.annotationparam.update_annotation(self)
    

def compute_center(x1, y1, x2, y2):
    return .5*(x1+x2), .5*(y1+y2)
    
def compute_rect_size(x1, y1, x2, y2):
    return x2-x1, fabs(y2-y1)

def compute_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)
    
def compute_angle(x1, y1, x2, y2, reverse=False):
    sign = -1 if reverse else 1
    return np.arctan(-sign*(y2-y1)/(x2-x1))*180/np.pi


class AnnotatedPoint(AnnotatedShape):
    """
    Construct an annotated point at coordinates (x, y) 
    with properties set with *annotationparam* 
    (see :py:class:`guiqwt.styles.AnnotationParam`)
    """
    SHAPE_CLASS = PointShape
    LABEL_ANCHOR = "TL"
    def __init__(self, x=0, y=0, annotationparam=None):
        AnnotatedShape.__init__(self, annotationparam)
        self.set_pos(x, y)
        
    #----Public API-------------------------------------------------------------
    def set_pos(self, x, y):
        """Set the point coordinates to (x, y)"""
        self.shape.set_pos(x, y)
        self.set_label_position()

    def get_pos(self):
        """Return the point coordinates"""
        return self.shape.get_pos()
        
    #----AnnotatedShape API-----------------------------------------------------
    def create_shape(self):
        """Return the shape object associated to this annotated shape object"""
        shape = self.SHAPE_CLASS(0, 0)
        return shape

    def set_label_position(self):
        """Set label position, for instance based on shape position"""
        x, y = self.shape.points[0]
        self.label.set_position(x, y)
        
    #----AnnotatedShape API-----------------------------------------------------
    def get_infos(self):
        """Return formatted string with informations on current shape"""
        f = self.annotationparam.format
        xt, yt = self.apply_transform_matrix(*self.shape.points[0])
        return ("( "+f+u" ; "+f+" )") % (xt, yt)
        

class AnnotatedSegment(AnnotatedShape):
    """
    Construct an annotated segment between coordinates (x1, y1) and 
    (x2, y2) with properties set with *annotationparam* 
    (see :py:class:`guiqwt.styles.AnnotationParam`)
    """
    SHAPE_CLASS = SegmentShape
    LABEL_ANCHOR = "C"
    def __init__(self, x1=0, y1=0, x2=0, y2=0, annotationparam=None):
        AnnotatedShape.__init__(self, annotationparam)
        self.set_rect(x1, y1, x2, y2)
        
    #----AnnotatedShape API-----------------------------------------------------
    def set_rect(self, x1, y1, x2, y2):
        """
        Set the coordinates of the shape's top-left corner to (x1, y1), 
        and of its bottom-right corner to (x2, y2).
        """
        self.shape.set_rect(x1, y1, x2, y2)
        self.set_label_position()

    def get_rect(self):
        """
        Return the coordinates of the shape's top-left and bottom-right corners
        """
        return self.shape.get_rect()
        
    def get_length(self):
        """Return segment length"""
        return compute_distance(*self.get_transformed_coords(0, 2))
    
    def set_label_position(self):
        """Set label position, for instance based on shape position"""
        x1, y1, x2, y2 = self.get_rect()
        self.label.set_position(*compute_center(x1, y1, x2, y2))
        
    #----AnnotatedShape API-----------------------------------------------------
    def get_infos(self):
        """Return formatted string with informations on current shape"""
        f = self.annotationparam.format
        return _("Distance:") + (" "+f) % self.get_length()


class AnnotatedRectangle(AnnotatedShape):
    """
    Construct an annotated rectangle between coordinates (x1, y1) and 
    (x2, y2) with properties set with *annotationparam* 
    (see :py:class:`guiqwt.styles.AnnotationParam`)
    """
    SHAPE_CLASS = RectangleShape
    LABEL_ANCHOR = "TL"
    def __init__(self, x1=0, y1=0, x2=0, y2=0, annotationparam=None):
        AnnotatedShape.__init__(self, annotationparam)
        self.set_rect(x1, y1, x2, y2)
        
    #----Public API-------------------------------------------------------------
    def set_rect(self, x1, y1, x2, y2):
        """
        Set the coordinates of the shape's top-left corner to (x1, y1), 
        and of its bottom-right corner to (x2, y2).
        """
        self.shape.set_rect(x1, y1, x2, y2)
        self.set_label_position()

    def get_rect(self):
        """
        Return the coordinates of the shape's top-left and bottom-right corners
        """
        return self.shape.get_rect()
        
    def get_center(self):
        """Return center coordinates: (xc, yc)"""
        return compute_center(*self.get_transformed_coords(0, 2))
        
    def get_size(self):
        """Return rectangle size: (width, height)"""
        return compute_rect_size(*self.get_transformed_coords(0, 2))
        
    #----AnnotatedShape API-----------------------------------------------------
    def set_label_position(self):
        """Set label position, for instance based on shape position"""
        x_label, y_label = self.shape.points.min(axis=0)
        self.label.set_position(x_label, y_label)
    
    def get_computations_text(self):
        """Return formatted string with informations on current shape"""
        tdict = self.get_string_dict()
        return u"%(center_n)s ( %(center)s )<br>%(size_n)s %(size)s" % tdict
        
    #----AnnotatedShape API-----------------------------------------------------
    def get_infos(self):
        """Return formatted string with informations on current shape"""
        f = self.annotationparam.format
        return "<br>".join([
                    _("Center:") + (" ( "+f+u" ; "+f+" )") % self.get_center(),
                    _("Size:") + (" "+f+u" x "+f) % self.get_size(),
                            ])


class AnnotatedEllipse(AnnotatedShape):
    """
    Construct an annotated ellipse with X-axis diameter between 
    coordinates (x1, y1) and (x2, y2) 
    and properties set with *annotationparam* 
    (see :py:class:`guiqwt.styles.AnnotationParam`)
    """
    SHAPE_CLASS = EllipseShape
    LABEL_ANCHOR = "C"
    def __init__(self, x1=0, y1=0, x2=0, y2=0, ratio=1., annotationparam=None):
        self.ratio = ratio
        AnnotatedShape.__init__(self, annotationparam)
        self.set_xdiameter(x1, y1, x2, y2)

    #----Public API-------------------------------------------------------------
    def set_xdiameter(self, x0, y0, x1, y1):
        """Set the coordinates of the ellipse's X-axis diameter"""
        self.shape.set_xdiameter(x0, y0, x1, y1)
        self.set_label_position()
                         
    def get_xdiameter(self):
        """Return the coordinates of the ellipse's X-axis diameter"""
        return self.shape.get_xdiameter()
                         
    def set_ydiameter(self, x2, y2, x3, y3):
        """Set the coordinates of the ellipse's Y-axis diameter"""
        self.shape.set_ydiameter(x2, y2, x3, y3)
        self.set_label_position()
                         
    def get_ydiameter(self):
        """Return the coordinates of the ellipse's Y-axis diameter"""
        return self.shape.get_ydiameter()

    def get_rect(self):
        return self.shape.get_rect()
    
    def set_rect(self, x0, y0, x1, y1):
        raise NotImplementedError
        
    def get_center(self):
        """Return center coordinates: (xc, yc)"""
        return compute_center(*self.get_transformed_coords(0, 1))
        
    def get_size(self):
        """Return ellipse size: (width, height)"""
        xcoords = self.get_transformed_coords(0, 1)
        ycoords = self.get_transformed_coords(2, 3)
        dx = compute_distance(*xcoords)
        dy = compute_distance(*ycoords)
        if fabs(self.get_angle()) > 45:
            dx, dy = dy, dx
        return dx, dy
        
    def get_angle(self):
        """Return X-diameter angle with horizontal direction"""
        xcoords = self.get_transformed_coords(0, 1)
        _x, yr1 = self.apply_transform_matrix(1., 1.)
        _x, yr2 = self.apply_transform_matrix(1., 2.)
        return (compute_angle(reverse=yr1 > yr2, *xcoords)+90)%180-90
        
    #----AnnotatedShape API-----------------------------------------------------
    def create_shape(self):
        """Return the shape object associated to this annotated shape object"""
        shape = self.SHAPE_CLASS(0, 0, 1, 1, ratio=self.ratio)
        return shape
        
    def set_label_position(self):
        """Set label position, for instance based on shape position"""
        x_label, y_label = self.shape.points.mean(axis=0)
        self.label.set_position(x_label, y_label)
        
    #----AnnotatedShape API-----------------------------------------------------
    def get_infos(self):
        """Return formatted string with informations on current shape"""
        f = self.annotationparam.format
        return "<br>".join([
                    _("Center:") + (" ( "+f+u" ; "+f+" )") % self.get_center(),
                    _("Size:") + (" "+f+u" x "+f) % self.get_size(),
                    _(u"Angle:") + u" %.1f°" % self.get_angle(),
                            ])
        

class AnnotatedCircle(AnnotatedEllipse):
    """
    Construct an annotated circle with diameter between coordinates 
    (x1, y1) and (x2, y2) and properties set with *annotationparam* 
    (see :py:class:`guiqwt.styles.AnnotationParam`)
    """
    def __init__(self, x1=0, y1=0, x2=0, y2=0, annotationparam=None):
        AnnotatedEllipse.__init__(self, x1, y1, x2, y2, 1., annotationparam)
        
    def get_diameter(self):
        """Return circle diameter"""
        return compute_distance(*self.get_transformed_coords(0, 1))
        
    #----AnnotatedShape API-------------------------------------------------
    def get_infos(self):
        """Return formatted string with informations on current shape"""
        f = self.annotationparam.format
        return "<br>".join([
                    _("Center:") + (" ( "+f+u" ; "+f+" )") % self.get_center(),
                    _("Diameter:") + (" "+f) % self.get_diameter(),
                            ])

    #----AnnotatedEllipse API---------------------------------------------------
    def set_rect(self, x0, y0, x1, y1):
        self.shape.set_rect(x0, y0, x1, y1)
        


class AnnotatedCursor(AnnotatedShape):
    """
    Construct an annotated vertical cursor in x=pos
    with properties set with *annotationparam* 
    (see :py:class:`guiqwt.styles.AnnotationParam`)
    """
    LABEL_ANCHOR = "BL"

    def __init__(self, pos=0, annotationparam=None, moveable=True):
        self._can_move = moveable
        self._can_resize = moveable
        
        AnnotatedShape.__init__(self, annotationparam)
        self.set_pos(pos)
        
    #----IBasePlotItem API------------------------------------------------------
    def select(self):
        """Select item"""
        AnnotatedShape.select(self)
        param = self.label.labelparam
        param.color = self.shape.shapeparam.sel_line.color
        param.update_label(self.label)
        
    def unselect(self):
        """Unselect item"""
        AnnotatedShape.unselect(self)
        param = self.label.labelparam
        param.color = self.shape.shapeparam.line.color
        param.update_label(self.label)
        
    #----AnnotatedShape API-----------------------------------------------------
    def set_pos(self, pos):
        self.shape.set_pos(pos, dosignal=False)

    def get_pos(self):
        return self.shape.get_pos()
        
    def set_style(self, section, option):
        pass
           
    def hit_test(self, pos):
        return self.shape.hit_test(pos)
        
    def move_point_to(self, handle, pos, ctrl=None):
        super(AnnotatedCursor, self).move_point_to(handle, pos, ctrl)
        if self.plot():
            self.plot().emit(SIG_CURSOR_MOVED, self, self.get_pos())
        
    def draw(self, painter, xMap, yMap, canvasRect):
        self.set_label_position()
        super(AnnotatedCursor, self).draw(painter, xMap, yMap, canvasRect)
        
    def create_label(self):
        """Return the label object associated to this annotated shape object"""
        label_param = LabelParam(_("Label"), icon='label.png')
        label_param.read_config(CONF, "plot", "shape/cursor_label")
        label_param.anchor = self.LABEL_ANCHOR
        label_param.color = self.shape.shapeparam.line.color
        return DataInfoLabel(label_param, [self])
        
    #----AnnotatedShape API-----------------------------------------------------
    def create_shape(self):
        """Return the shape object associated to this annotated shape object"""
        return self.SHAPE_CLASS(0, self.can_move())
        
    def get_infos(self):
        """Return dictionary with measured data on shape"""
        f = self.annotationparam.format
        return f % self.get_pos()
        
        
class AnnotatedVCursor(AnnotatedCursor):
    SHAPE_CLASS = VerticalCursor
    def move_local_point_to(self, handle, pos, ctrl=None):
        val = self.plot().invTransform(self.xAxis(), pos.x())
        self.move_point_to(handle, (val, 0))
        
    def set_label_position(self):
        """Set label position, for instance based on shape position"""
        plot = self.plot()
        y = plot.invTransform(self.yAxis(),
                              plot.canvas().contentsRect().bottomLeft().y())
        self.label.set_position(self.shape.pos, y)
        

class AnnotatedHCursor(AnnotatedCursor):
    SHAPE_CLASS = HorizontalCursor
    def move_local_point_to(self, handle, pos, ctrl=None):
        val = self.plot().invTransform(self.yAxis(), pos.y())
        self.move_point_to(handle, (val, 0))
        
    def set_label_position(self):
        """Set label position, for instance based on shape position"""
        plot = self.plot()
        x = plot.invTransform(self.xAxis(),
                              plot.canvas().contentsRect().bottomLeft().x())
        self.label.set_position(x, self.shape.pos)
