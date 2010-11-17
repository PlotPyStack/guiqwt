# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
Ready-to-use curve and image plotting dialog boxes
"""

import weakref
from PyQt4.QtGui import (QDialogButtonBox, QVBoxLayout, QGridLayout, QToolBar,
                         QDialog, QHBoxLayout, QMenu, QActionGroup, QSplitter,
                         QSizePolicy, QApplication)
from PyQt4.QtCore import Qt, SIGNAL, SLOT

from guidata.configtools import get_icon
from guidata.utils import assert_interfaces_valid
from guidata.qthelpers import create_action

# Local imports
from guiqwt.config import _
from guiqwt.baseplot import EnhancedQwtPlot
from guiqwt.curve import CurvePlot, PlotItemList
from guiqwt.image import ImagePlot
from guiqwt.tools import (SelectTool, RectZoomTool, ColormapTool,
                          ReverseYAxisTool, BasePlotMenuTool, HelpTool,
                          ItemListTool, AntiAliasingTool, PrintTool,
                          DisplayCoordsTool, AxisScaleTool, SaveAsTool,
                          AspectRatioTool, ContrastTool, DummySeparatorTool,
                          XCrossSectionTool, YCrossSectionTool, SnapshotTool,
                          AverageCrossSectionsTool)
from guiqwt.interfaces import IPlotManager
from guiqwt.signals import (SIG_ITEMS_CHANGED, SIG_ACTIVE_ITEM_CHANGED,
                            SIG_VISIBILITY_CHANGED)
from guiqwt.panels import (ITEMLIST_PANEL_ID, CONTRAST_PANEL_ID,
                           XCS_PANEL_ID, YCS_PANEL_ID)


class PlotManager(object):
    """
    A 'controller' that organizes relations between
    plots (EnhancedQwtPlot), panels, tools (GuiTool) and toolbar
    """
    __implements__ = (IPlotManager,)

    def __init__(self, main):
        self.main = main # The main parent widget
        self.plots = {} # maps ids to instances of EnhancedQwtPlot
        self.panels = {} # Qt widgets that need to know about the plots
        self.tools = []
        self.toolbars = {}
        self.active_tool = None
        self.default_tool = None
        self.default_plot = None
        self.default_toolbar = None
        self.groups = {} # Action groups for grouping QActions

    def add_plot(self, plot, id):
        assert id not in self.plots
        assert isinstance(plot, EnhancedQwtPlot)
        assert not self.tools, "tools must be added after plots"
        assert not self.panels, "panels must be added after plots"
        self.plots[id] = plot
        if len(self.plots) == 1:
            self.default_plot = plot
        plot.set_manager(self)
        # Connecting signals
        plot.connect(plot, SIG_ITEMS_CHANGED, self.update_tools_status)
        plot.connect(plot, SIG_ACTIVE_ITEM_CHANGED, self.update_tools_status)
        
    def set_default_plot(self, plot):
        self.default_plot = plot
        
    def get_default_plot(self):
        return self.default_plot

    def add_panel(self, panel):
        assert panel.PANEL_ID not in self.panels
        assert not self.tools, "tools must be added after panels"
        self.panels[panel.PANEL_ID] = panel
        panel.register_panel(self)

    def add_toolbar(self, toolbar, id):
        assert id not in self.toolbars
        self.toolbars[id] = toolbar
        if self.default_toolbar is None:
            self.default_toolbar = toolbar
            
    def set_default_toolbar(self, toolbar):
        self.default_toolbar = toolbar
        
    def get_default_toolbar(self):
        return self.default_toolbar

    def add_tool(self, ToolKlass, *args, **kwargs):
        tool = ToolKlass(self, *args, **kwargs)
        self.tools.append(tool)
        for plot in self.plots.values():
            tool.register_plot(plot)
        if len(self.tools) == 1:
            self.default_tool = tool
        return tool
        
    def add_separator_tool(self, toolbar_id=None):
        if toolbar_id is None:
            for _id, toolbar in self.toolbars.iteritems():
                if toolbar is self.get_default_toolbar():
                    toolbar_id = _id
                    break
        self.add_tool(DummySeparatorTool, toolbar_id)
        
    def set_default_tool(self, tool):
        self.default_tool = tool

    def get_default_tool(self):
        return self.default_tool

    def get_active_tool(self):
        return self.active_tool

    def set_active_tool(self, tool=None):
        """Activate tool or default tool"""
        self.active_tool = tool

    def get_plot(self, id=None):
        if id is None:
            return self.default_plot
        return self.plots[id]

    def get_plots(self):
        return self.plots.values()

    def get_active_plot(self):
        """The active plot is the plot whose canvas has the focus
        otherwise it's the "default" plot
        """
        for plot in self.plots.values():
            canvas = plot.canvas()
            if canvas.hasFocus():
                return plot
        return self.default_plot

    def get_tool_group(self, groupname):
        group = self.groups.get(groupname, None)
        if group is None:
            group = QActionGroup(self.main)
            self.groups[groupname] = weakref.ref(group)
            return group
        else:
            return group()

    def get_main(self):
        return self.main

    def get_panel(self, panel_id):
        return self.panels.get(panel_id, None)

    def get_toolbar(self, tbname):
        return self.toolbars.get(tbname, None)

    def get_context_menu(self, plot):
        """Return widget context menu -- built using active tools"""
        menu = QMenu(plot)
        self.update_tools_status(plot)
        for tool in self.tools:
            tool.setup_context_menu(menu, plot)
        return menu
        
    def update_tools_status(self, plot):
        """Update tools for current plot"""
        for tool in self.tools:
            tool.update_status(plot)

    def create_action(self, title, triggered=None, toggled=None,
                      shortcut=None, icon=None, tip=None):
        return create_action(self.main, title, triggered=triggered,
                             toggled=toggled, shortcut=shortcut,
                             icon=icon, tip=tip)

    # The following methods provide some sets of tools that
    # are often registered together
    def register_standard_tools(self):
        """Registering basic tools for standard plot dialog
        --> top of the context-menu"""
        t = self.add_tool(SelectTool)
        self.set_default_tool(t)
        self.add_tool(RectZoomTool)
        self.add_tool(BasePlotMenuTool, "item")
        self.add_tool(BasePlotMenuTool, "grid")
        self.add_tool(BasePlotMenuTool, "axes")
        self.add_tool(DisplayCoordsTool)
        if self.get_panel(ITEMLIST_PANEL_ID):
            self.add_tool(ItemListTool)

    def register_curve_tools(self):
        """Registering specific tools
        --> middle of the context-menu"""
        self.add_tool(AntiAliasingTool)
        self.add_tool(AxisScaleTool)
        
    def register_other_tools(self):
        """Registering other common tools
        --> bottom of the context-menu"""
        self.add_tool(SaveAsTool)
        self.add_tool(PrintTool)
        self.add_tool(HelpTool)

    def register_image_tools(self):
        self.add_tool(ColormapTool)
        self.add_tool(ReverseYAxisTool)
        self.add_tool(AspectRatioTool)
        if self.get_panel(CONTRAST_PANEL_ID):
            self.add_tool(ContrastTool)
        if self.get_panel(XCS_PANEL_ID) and self.get_panel(YCS_PANEL_ID):
            self.add_tool(XCrossSectionTool)
            self.add_tool(YCrossSectionTool)
            self.add_tool(AverageCrossSectionsTool)
        self.add_tool(SnapshotTool)

assert_interfaces_valid(PlotManager)


#===============================================================================
# Curve Plot Widget/Dialog with integrated Item list widget
#===============================================================================
def configure_plot_splitter(qsplit, decreasing_size=True):
    qsplit.setChildrenCollapsible(False)
    qsplit.setHandleWidth(4)
    if decreasing_size:
        qsplit.setStretchFactor(0, 1)
        qsplit.setStretchFactor(1, 0)
        qsplit.setSizes([2, 1])
    else:
        qsplit.setStretchFactor(0, 0)
        qsplit.setStretchFactor(1, 1)
        qsplit.setSizes([1, 2])

class CurvePlotWidget(QSplitter):
    """
    CurvePlotWidget
    
    parent: parent widget
    title: plot title
    xlabel: (bottom axis title, top axis title) or bottom axis title only
    ylabel: (left axis title, right axis title) or left axis title only
    """
    def __init__(self, parent=None, title=None, xlabel=None, ylabel=None,
                 section="plot", show_itemlist=False, gridparam=None):
        super(CurvePlotWidget, self).__init__(Qt.Horizontal, parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.plot = CurvePlot(parent=self,
                              title=title, xlabel=xlabel, ylabel=ylabel,
                              section=section, gridparam=gridparam)
        self.addWidget(self.plot)
        self.itemlist = PlotItemList(self)
        self.itemlist.setVisible(show_itemlist)
        self.addWidget(self.itemlist)
        configure_plot_splitter(self)
        
        self.manager = PlotManager(self)
        self.manager.add_plot(self.plot, id(self.plot))
        self.manager.add_panel(self.itemlist)
    
    def register_tool(self, Klass, *args, **kwargs):
        return self.manager.add_tool(Klass, *args, **kwargs)

    def register_tools(self):
        """Derived classes can override this method
        to provide a fully customized set of tools"""
        self.manager.register_standard_tools()
        self.manager.add_separator_tool()
        self.manager.register_curve_tools()
        self.manager.add_separator_tool()
        self.manager.register_other_tools()
        self.activate_default_tool()
        
    def activate_default_tool(self):
        self.manager.get_default_tool().activate()
        
    def get_plot(self):
        """Return CurvePlot/ImagePlot instance"""
        return self.manager.get_plot()

    def get_panel(self, id):
        """Return panel associated to *id*"""
        return self.manager.get_panel(id)


class CurvePlotDialog(QDialog):
    def __init__(self, wintitle="guiqwt plot", icon="guiqwt.png",
                 edit=False, toolbar=False, options=None, parent=None):
        super(CurvePlotDialog, self).__init__(parent)
        self.edit = edit
        self.setWindowTitle(wintitle)
        if isinstance(icon, basestring):
            icon = get_icon(icon)
        self.setWindowIcon(icon)
        self.setMinimumSize(320, 240)
        self.resize(640, 480)
        self.setWindowFlags(Qt.Window)
        
        self.layout = QGridLayout()
        
        if options is None:
            options = {}
            
        self.plotwidget = None
        self.create_plot(options)
        self.manager = self.plotwidget.manager
        
        self.vlayout = QVBoxLayout(self)
        
        self.toolbar = QToolBar(_("Tools"))
        self.manager.add_toolbar(self.toolbar, "default")
        if not toolbar:
            self.toolbar.hide()
        self.vlayout.addWidget(self.toolbar)
        
        self.setLayout(self.vlayout)
        self.vlayout.addLayout(self.layout)
        
        if self.edit:
            self.button_layout = QHBoxLayout()
            self.install_button_layout()
            self.vlayout.addLayout(self.button_layout)
        
        self.register_tools()
        
    def install_button_layout(self):
        """Install standard buttons (OK, Cancel) for dialog box
        May be overriden to customize button box"""
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.connect(bbox, SIGNAL("accepted()"), SLOT("accept()"))
        self.connect(bbox, SIGNAL("rejected()"), SLOT("reject()"))
        self.button_layout.addWidget(bbox)
    
    def register_tool(self, Klass, *args, **kwargs):
        return self.plotwidget.register_tool(Klass, *args, **kwargs)

    def register_tools(self):
        """Derived classes can override this method
        to provide a fully customized set of tools"""
        self.plotwidget.register_tools()
        
    def activate_default_tool(self):
        self.plotwidget.activate_default_tool()

    def create_plot(self, options):
        """CurvePlotWidget instantiation
        May be overriden to customize plot layout"""
        self.plotwidget = CurvePlotWidget(self, **options)
        self.layout.addWidget(self.plotwidget, 0, 0)
        
    def get_plot(self):
        """Return CurvePlot/ImagePlot instance"""
        return self.manager.get_plot()

    def get_panel(self, id):
        """Return panel associated to *id*"""
        return self.manager.get_panel(id)


#===============================================================================
# Image Plot Widget/Dialog with integrated Levels Histogram and other widgets
#===============================================================================
class ImagePlotWidget(QSplitter):
    """
    ImagePlotWidget
    
    parent: parent widget
    title: plot title (string)
    xlabel, ylabel, zlabel: resp. bottom, left and right axis titles (strings)
    yreverse: reversing Y-axis (bool)
    aspect_ratio: height to width ratio (float)
    lock_aspect_ratio: locking aspect ratio (bool)
    show_contrast: showing contrast adjustment tool (bool)
    show_xsection: showing x-axis cross section plot (bool)
    show_ysection: showing y-axis cross section plot (bool)
    xsection_pos: x-axis cross section plot position (string: "top", "bottom")
    ysection_pos: y-axis cross section plot position (string: "left", "right")
    """
    def __init__(self, parent=None, title="",
                 xlabel=("", ""), ylabel=("", ""), zlabel=None, yreverse=True,
                 colormap="jet", aspect_ratio=1.0, lock_aspect_ratio=True,
                 show_contrast=False, show_itemlist=False, show_xsection=False,
                 show_ysection=False, xsection_pos="top", ysection_pos="right",
                 gridparam=None):
        super(ImagePlotWidget, self).__init__(Qt.Vertical, parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.sub_splitter = QSplitter(Qt.Horizontal, self)
        self.plot = ImagePlot(parent=self, title=title,
                              xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                              yreverse=yreverse, aspect_ratio=aspect_ratio,
                              lock_aspect_ratio=lock_aspect_ratio,
                              gridparam=gridparam)

        from guiqwt.cross_section import YCrossSectionWidget
        self.ycsw = YCrossSectionWidget(self, position=ysection_pos)
        self.ycsw.setVisible(show_ysection)
        
        from guiqwt.cross_section import XCrossSectionWidget
        self.xcsw = XCrossSectionWidget(self)
        self.xcsw.setVisible(show_xsection)
        
        self.connect(self.xcsw, SIG_VISIBILITY_CHANGED, self.xcsw_is_visible)
        
        xcsw_splitter = QSplitter(Qt.Vertical, self)
        if xsection_pos == "top":
            self.ycsw_spacer = self.ycsw.spacer1
            xcsw_splitter.addWidget(self.xcsw)
            xcsw_splitter.addWidget(self.plot)
        else:
            self.ycsw_spacer = self.ycsw.spacer2
            xcsw_splitter.addWidget(self.plot)
            xcsw_splitter.addWidget(self.xcsw)
        self.connect(xcsw_splitter, SIGNAL('splitterMoved(int,int)'),
                     lambda pos, index: self.adjust_ycsw_height())
        
        ycsw_splitter = QSplitter(Qt.Horizontal, self)
        if ysection_pos == "left":
            ycsw_splitter.addWidget(self.ycsw)
            ycsw_splitter.addWidget(xcsw_splitter)
        else:
            ycsw_splitter.addWidget(xcsw_splitter)
            ycsw_splitter.addWidget(self.ycsw)
            
        configure_plot_splitter(xcsw_splitter,
                                decreasing_size=xsection_pos == "bottom")
        configure_plot_splitter(ycsw_splitter,
                                decreasing_size=ysection_pos == "right")
        
        self.sub_splitter.addWidget(ycsw_splitter)
        
        self.itemlist = PlotItemList(self)
        self.itemlist.setVisible(show_itemlist)
        self.sub_splitter.addWidget(self.itemlist)
        
        # Contrast adjustment (Levels histogram)
        from guiqwt.histogram import ContrastAdjustment
        self.contrast = ContrastAdjustment(self)
        self.contrast.setVisible(show_contrast)
        self.addWidget(self.contrast)
        
        configure_plot_splitter(self)
        configure_plot_splitter(self.sub_splitter)
        
        self.manager = PlotManager(self)
        self.manager.add_plot(self.plot, id(self.plot))
        self.manager.add_panel(self.itemlist)
        self.manager.add_panel(self.xcsw)
        self.manager.add_panel(self.ycsw)
        self.manager.add_panel(self.contrast)
        
    def adjust_ycsw_height(self, height=None):
        if height is None:
            height = self.xcsw.height()-self.ycsw.toolbar.height()
        self.ycsw_spacer.changeSize(0, height,
                                    QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.ycsw.layout().invalidate()
        if height:
            QApplication.processEvents()
        
    def xcsw_is_visible(self, state):
        if state:
            QApplication.processEvents()
            self.adjust_ycsw_height()
        else:
            self.adjust_ycsw_height(0)

    def register_tool(self, Klass, *args, **kwargs):
        return self.manager.add_tool(Klass, *args, **kwargs)

    def register_tools(self):
        """Derived classes can override this method
        to provide a fully customized set of tools"""
        self.manager.register_standard_tools()
        self.manager.add_separator_tool()
        self.manager.register_image_tools()
        self.manager.add_separator_tool()
        self.manager.register_other_tools()
        self.activate_default_tool()
        
    def activate_default_tool(self):
        self.manager.get_default_tool().activate()
        
    def get_plot(self):
        """Return CurvePlot/ImagePlot instance"""
        return self.manager.get_plot()

    def get_panel(self, id):
        """Return panel associated to *id*"""
        return self.manager.get_panel(id)


class ImagePlotDialog(CurvePlotDialog):
    def __init__(self, wintitle="guiqwt imshow", icon="guiqwt.png",
                 edit=False, toolbar=False, options=None, parent=None):
        super(ImagePlotDialog, self).__init__(wintitle=wintitle, icon=icon,
                                              edit=edit, toolbar=toolbar,
                                              options=options, parent=parent)

    def create_plot(self, options, row=0, column=0, rowspan=1, columnspan=1):
        self.plotwidget = ImagePlotWidget(self, **options)
        self.layout.addWidget(self.plotwidget, row, column, rowspan, columnspan)
