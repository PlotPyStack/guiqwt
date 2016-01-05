# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)
"""
Dot array example
=================

Example showing how to create a custom item (drawing dots of variable size) 
and integrate the associated `guidata` dataset (GUI-based form) to edit its 
parameters (directly into the same window as the plot itself, *and* within 
the custom item parameters: right-click on the selectable item to open the 
associated dialog box).
"""

SHOW = True # Show test in GUI-based test launcher

from guidata.qt.QtCore import QPointF, QRectF
from guidata.qt.QtGui import QPen, QBrush, QColor, QMessageBox, QPainter

from guidata.dataset.datatypes import DataSet, BeginGroup, EndGroup
from guidata.dataset.dataitems import FloatItem, ColorItem
from guidata.dataset.qtwidgets import DataSetEditGroupBox

from guiqwt.plot import ImageDialog
from guiqwt.curve import vmap
from guiqwt.image import RawImageItem, IImageItemType
from guiqwt.tools import SaveAsTool, CopyToClipboardTool, PrintTool, HelpTool


class DotArrayParam(DataSet):
    """Dot array"""
    g1 = BeginGroup("Size of the area")
    dim_h = FloatItem("Width", default=20, min=0, unit="mm")
    dim_v = FloatItem("Height", default=20, min=0, unit="mm")
    _g1 = EndGroup("Size of the area")

    g2 = BeginGroup("Grid pattern properties")
    step_x = FloatItem("Step in X-axis", default=1, min=1, unit="mm")
    step_y = FloatItem("Step in Y-axis", default=1, min=1, unit="mm")
    size = FloatItem("Dot size", default=.2, min=0, max=2, slider=True, unit="mm")
    color = ColorItem("Dot color", default="red")
    _g2 = EndGroup("Grid pattern properties")
    
    def update_image(self, obj):
        self._update_cb()
    
    def update_param(self, obj):
        pass


class DotArrayItem(RawImageItem):
    def __init__(self, imageparam=None):
        super(DotArrayItem, self).__init__(np.zeros((1, 1)), imageparam)
        self.update_border()

    def boundingRect(self):
        param = self.imageparam
        if param is not None:
            return QRectF(QPointF(0, 0),
                          QPointF(param.dim_h+param.size,
                                  param.dim_v+param.size))

    def types(self):
        return (IImageItemType,)

    def draw_image(self, painter, canvasRect, srcRect, dstRect, xMap, yMap): 
        painter.setRenderHint(QPainter.Antialiasing, True)
        param = self.imageparam
        xcoords = vmap(xMap, np.arange(0, param.dim_h+1, param.step_x))
        ycoords = vmap(yMap, np.arange(0, param.dim_v+1, param.step_y))
        rx = .5*param.size*xMap.pDist()/xMap.sDist()
        ry = .5*param.size*yMap.pDist()/yMap.sDist()
        color = QColor(param.color)
        painter.setPen(QPen(color))
        painter.setBrush(QBrush(color))
        for xc in xcoords:
            for yc in ycoords:
                painter.drawEllipse(QPointF(xc, yc), rx, ry)


class CustomHelpTool(HelpTool):
    def activate_command(self, plot, checked):
        QMessageBox.information(plot, "Help",
                                """**to be customized**
Keyboard/mouse shortcuts:
  - single left-click: item (curve, image, ...) selection
  - single right-click: context-menu relative to selected item
  - shift: on-active-curve (or image) cursor
  - alt: free cursor
  - left-click + mouse move: move item (when available)
  - middle-click + mouse move: pan
  - right-click + mouse move: zoom""")


class DotArrayDialog(ImageDialog):
    def __init__(self):
        self.item = None
        self.stamp_gbox = None
        super(DotArrayDialog, self).__init__(wintitle="Dot array example",
#            icon="path/to/your_icon.png",
            toolbar=True, edit=True)        
        self.resize(900, 600)
        
    def register_tools(self):
        self.register_standard_tools()
        self.add_separator_tool()
        self.add_tool(SaveAsTool)
        self.add_tool(CopyToClipboardTool)
        self.add_tool(PrintTool)
        self.add_tool(CustomHelpTool)
        self.activate_default_tool()
        plot = self.get_plot()
        plot.enableAxis(plot.yRight, False)
        plot.set_aspect_ratio(lock=True)

    def create_plot(self, options):
        self.stamp_gbox = DataSetEditGroupBox("Dots", DotArrayParam)
        try:
            # guiqwt v3:
            self.stamp_gbox.SIG_APPLY_BUTTON_CLICKED.connect(self.apply_params)
        except AttributeError:
            # guiqwt v2:
            from guidata.qt.QtCore import SIGNAL
            self.connect(self.stamp_gbox, SIGNAL("apply_button_clicked()"),
                         self.apply_params)
        self.plot_layout.addWidget(self.stamp_gbox, 0, 1)
        options = dict(title="Main plot")
        ImageDialog.create_plot(self, options, 0, 0, 2, 1)
        
    def show_data(self, param):
        plot = self.get_plot()
        if self.item is None:
            param._update_cb = lambda: self.stamp_gbox.get()
            self.item = DotArrayItem(param)
            plot.add_item(self.item)
        plot.do_autoscale()
        
    def apply_params(self):
        param = self.stamp_gbox.dataset
        self.show_data(param)

        
if __name__ == "__main__":
    # -- Create QApplication
    import guidata
    import numpy as np
    _app = guidata.qapplication()
    
    dlg = DotArrayDialog()
    dlg.apply_params()
    dlg.exec_()
