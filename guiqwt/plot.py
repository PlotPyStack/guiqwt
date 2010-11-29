# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.plot
-----------

The `plot` module provides the following features:
    * :py:class:`guiqwt.plot.PlotManager`: the `plot manager` is an object to 
      link `plots`, `panels` and `tools` together for designing highly 
      versatile graphical user interfaces
    * :py:class:`guiqwt.plot.CurvePlotWidget`: a ready-to-use widget for curve 
      displaying with an integrated and preconfigured `plot manager` providing 
      the `item list panel` and curve-related `tools`
    * :py:class:`guiqwt.plot.CurvePlotDialog`: a ready-to-use dialog box for 
      curve displaying with an integrated and preconfigured `plot manager` 
      providing the `item list panel` and curve-related `tools`
    * :py:class:`guiqwt.plot.ImagePlotWidget`: a ready-to-use widget for curve 
      and image displaying with an integrated and preconfigured `plot manager` 
      providing the `item list panel`, the `contrast adjustment` panel, the 
      `cross section panels` (along X and Y axes) and image-related `tools` 
      (e.g. colormap selection tool)
    * :py:class:`guiqwt.plot.ImagePlotDialog`: a ready-to-use dialog box for 
      curve and image displaying with an integrated and preconfigured 
      `plot manager` providing the `item list panel`, the `contrast adjustment` 
      panel, the `cross section panels` (along X and Y axes) and image-related 
      `tools` (e.g. colormap selection tool)

.. seealso::
    
    Module :py:mod:`guiqwt.curve`
        Module providing curve-related plot items and plotting widgets
        
    Module :py:mod:`guiqwt.image`
        Module providing image-related plot items and plotting widgets
        
    Module :py:mod:`guiqwt.tools`
        Module providing the `plot tools`
        
    Module :py:mod:`guiqwt.panels`
        Module providing the `plot panels` IDs
        
    Module :py:mod:`guiqwt.signals`
        Module providing all the end-user Qt SIGNAL objects defined in `guiqwt`
        
    Module :py:mod:`guiqwt.baseplot`
        Module providing the `guiqwt` plotting widget base class

Examples
~~~~~~~~

Simple example *without* the `plot manager`:

.. literalinclude:: ../guiqwt/tests/filtertest1.py
   :start-after: SHOW

Simple example *with* the `plot manager`:
even if this simple example does not justify the use of the `plot manager` 
(this is an unnecessary complication here), it shows how to use it. In more 
complex applications, using the `plot manager` allows to design highly versatile
graphical user interfaces.

.. literalinclude:: ../guiqwt/tests/filtertest2.py
   :start-after: SHOW

Reference
~~~~~~~~~

.. autoclass:: PlotManager
   :members:
   :inherited-members:
.. autoclass:: CurvePlotWidget
   :members:
.. autoclass:: CurvePlotDialog
   :members:
.. autoclass:: ImagePlotWidget
   :members:
.. autoclass:: ImagePlotDialog
   :members:
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


class DefaultPlotID(object):
    pass

class PlotManager(object):
    """
    Construct a PlotManager object, a 'controller' that organizes relations 
    between plots (i.e. :py:class:`guiqwt.curve.CurvePlot` or 
    :py:class:`guiqwt.image.ImagePlot` objects), panels, 
    tools (see :py:mod:`guiqwt.tools`) and toolbars
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

    def add_plot(self, plot, plot_id=DefaultPlotID):
        """
        Register a plot to the plot manager:
            * plot: :py:class:`guiqwt.curve.CurvePlot` or 
              :py:class:`guiqwt.image.ImagePlot` object
            * plot_id (default id is the plot object's id: ``id(plot)``): 
              unique ID identifying the plot (any Python object), 
              this ID will be asked by the manager to access this plot later.
        
        Plot manager's registration sequence is the following:
            1. add plots
            2. add panels
            3. add tools
        """
        if plot_id is DefaultPlotID:
            plot_id = id(plot)
        assert plot_id not in self.plots
        assert isinstance(plot, EnhancedQwtPlot)
        assert not self.tools, "tools must be added after plots"
        assert not self.panels, "panels must be added after plots"
        self.plots[plot_id] = plot
        if len(self.plots) == 1:
            self.default_plot = plot
        plot.set_manager(self)
        # Connecting signals
        plot.connect(plot, SIG_ITEMS_CHANGED, self.update_tools_status)
        plot.connect(plot, SIG_ACTIVE_ITEM_CHANGED, self.update_tools_status)
        
    def set_default_plot(self, plot):
        """
        Set default plot
        
        The default plot is the plot on which tools and panels will act.
        """
        self.default_plot = plot
        
    def get_default_plot(self):
        """
        Return default plot
        
        The default plot is the plot on which tools and panels will act.
        """
        return self.default_plot

    def add_panel(self, panel):
        """
        Register a panel to the plot manager
        
        Plot manager's registration sequence is the following:
            1. add plots
            2. add panels
            3. add tools
        """
        assert panel.PANEL_ID not in self.panels
        assert not self.tools, "tools must be added after panels"
        self.panels[panel.PANEL_ID] = panel
        panel.register_panel(self)

    def add_toolbar(self, toolbar, toolbar_id="default"):
        """
        Add toolbar to the plot manager
            toolbar: a QToolBar object
            toolbar_id: toolbar's id (default id is string "default")
        """
        assert toolbar_id not in self.toolbars
        self.toolbars[toolbar_id] = toolbar
        if self.default_toolbar is None:
            self.default_toolbar = toolbar
            
    def set_default_toolbar(self, toolbar):
        """
        Set default toolbar
        """
        self.default_toolbar = toolbar
        
    def get_default_toolbar(self):
        """
        Return default toolbar
        """
        return self.default_toolbar

    def add_tool(self, ToolKlass, *args, **kwargs):
        """
        Register a tool to the manager
            * ToolKlass: tool's class (`guiqwt` builtin tools are defined in 
              module :py:mod:`guiqwt.tools`)
            * *args: arguments sent to the tool's class
            * **kwargs: keyword arguments sent to the tool's class
        
        Plot manager's registration sequence is the following:
            1. add plots
            2. add panels
            3. add tools
        """
        tool = ToolKlass(self, *args, **kwargs)
        self.tools.append(tool)
        for plot in self.plots.values():
            tool.register_plot(plot)
        if len(self.tools) == 1:
            self.default_tool = tool
        return tool
        
    def add_separator_tool(self, toolbar_id=None):
        """
        Register a separator tool to the plot manager: the separator tool is 
        just a tool which insert a separator in the plot context menu
        """
        if toolbar_id is None:
            for _id, toolbar in self.toolbars.iteritems():
                if toolbar is self.get_default_toolbar():
                    toolbar_id = _id
                    break
        self.add_tool(DummySeparatorTool, toolbar_id)
        
    def set_default_tool(self, tool):
        """
        Set default tool
        """
        self.default_tool = tool

    def get_default_tool(self):
        """
        Get default tool
        """
        return self.default_tool

    def activate_default_tool(self):
        """
        Activate default tool
        """
        self.get_default_tool().activate()

    def get_active_tool(self):
        """
        Return active tool
        """
        return self.active_tool

    def set_active_tool(self, tool=None):
        """
        Set active tool (if tool argument is None, the active tool will be 
        the default tool)
        """
        self.active_tool = tool

    def get_plot(self, plot_id=DefaultPlotID):
        """
        Return plot associated to `plot_id` (if method is called without 
        specifying the `plot_id` parameter, return the default plot)
        """
        if plot_id is DefaultPlotID:
            return self.default_plot
        return self.plots[plot_id]

    def get_plots(self):
        """
        Return all registered plots
        """
        return self.plots.values()

    def get_active_plot(self):
        """
        Return the active plot
        
        The active plot is the plot whose canvas has the focus
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
        """
        Return the main (parent) widget
        
        Note that for py:class:`guiqwt.plot.CurvePlotWidget` or 
        :py:class:`guiqwt.plot.ImagePlotWidget` objects, this method will 
        return the widget itself because the plot manager is integrated to it.
        """
        return self.main

    def get_panel(self, panel_id):
        """
        Return panel from its ID
        Panel IDs are listed in module guiqwt.panels
        """
        return self.panels.get(panel_id, None)
        
    def get_itemlist_panel(self):
        """
        Convenience function to get the `item list panel`
        
        Return None if the item list panel has not been added to this manager
        """
        from guiqwt import panels
        return self.get_panel(panels.ID_ITEMLIST)
        
    def get_contrast_panel(self):
        """
        Convenience function to get the `contrast adjustment panel`
        
        Return None if the contrast adjustment panel has not been added 
        to this manager
        """
        from guiqwt import panels
        return self.get_panel(panels.ID_CONTRAST)
        
    def get_xcs_panel(self):
        """
        Convenience function to get the `X-axis cross section panel`
        
        Return None if the X-axis cross section panel has not been added 
        to this manager
        """
        from guiqwt import panels
        return self.get_panel(panels.ID_XCS)
        
    def get_ycs_panel(self):
        """
        Convenience function to get the `Y-axis cross section panel`
        
        Return None if the Y-axis cross section panel has not been added 
        to this manager
        """
        from guiqwt import panels
        return self.get_panel(panels.ID_YCS)

    def get_toolbar(self, toolbar_id="default"):
        """
        Return toolbar from its ID
            toolbar_id: toolbar's id (default id is string "default")
        """
        return self.toolbars.get(toolbar_id, None)

    def get_context_menu(self, plot):
        """
        Return widget context menu -- built using active tools
        """
        menu = QMenu(plot)
        self.update_tools_status(plot)
        for tool in self.tools:
            tool.setup_context_menu(menu, plot)
        return menu
        
    def update_tools_status(self, plot):
        """
        Update tools for current plot
        """
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
        """
        Registering basic tools for standard plot dialog
        --> top of the context-menu
        """
        t = self.add_tool(SelectTool)
        self.set_default_tool(t)
        self.add_tool(RectZoomTool)
        self.add_tool(BasePlotMenuTool, "item")
        self.add_tool(BasePlotMenuTool, "grid")
        self.add_tool(BasePlotMenuTool, "axes")
        self.add_tool(DisplayCoordsTool)
        if self.get_itemlist_panel():
            self.add_tool(ItemListTool)

    def register_only_curve_tools(self):
        """
        Register only curve-related tools
        
        .. seealso:: methods 
        :py:meth:`guiqwt.plot.PlotManager.add_tool`
        :py:meth:`guiqwt.plot.PlotManager.register_standard_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_other_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_only_image_tools`
        """
        self.add_tool(AntiAliasingTool)
        self.add_tool(AxisScaleTool)
        
    def register_other_tools(self):
        """
        Register other common tools
        
        .. seealso:: methods 
        :py:meth:`guiqwt.plot.PlotManager.add_tool`
        :py:meth:`guiqwt.plot.PlotManager.register_standard_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_only_curve_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_only_image_tools`
        """
        self.add_tool(SaveAsTool)
        self.add_tool(PrintTool)
        self.add_tool(HelpTool)

    def register_only_image_tools(self):
        """
        Register only image-related tools
        
        .. seealso:: methods 
        :py:meth:`guiqwt.plot.PlotManager.add_tool`
        :py:meth:`guiqwt.plot.PlotManager.register_standard_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_other_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_only_curve_tools`
        """
        self.add_tool(ColormapTool)
        self.add_tool(ReverseYAxisTool)
        self.add_tool(AspectRatioTool)
        if self.get_contrast_panel():
            self.add_tool(ContrastTool)
        if self.get_xcs_panel() and self.get_ycs_panel():
            self.add_tool(XCrossSectionTool)
            self.add_tool(YCrossSectionTool)
            self.add_tool(AverageCrossSectionsTool)
        self.add_tool(SnapshotTool)
        
    def register_curve_tools(self):
        """
        Register standard, curve-related and other tools
        
        .. seealso:: methods 
        :py:meth:`guiqwt.plot.PlotManager.add_tool`
        :py:meth:`guiqwt.plot.PlotManager.register_standard_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_other_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_only_curve_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_only_image_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_image_tools`
        """
        self.register_standard_tools()
        self.add_separator_tool()
        self.register_only_curve_tools()
        self.add_separator_tool()
        self.register_other_tools()
        self.get_default_tool().activate()
    
    def register_image_tools(self):
        """
        Register standard, image-related and other tools
        
        .. seealso:: methods 
        :py:meth:`guiqwt.plot.PlotManager.add_tool`
        :py:meth:`guiqwt.plot.PlotManager.register_standard_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_other_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_only_curve_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_only_image_tools`
        :py:meth:`guiqwt.plot.PlotManager.register_curve_tools`
        """
        self.register_standard_tools()
        self.add_separator_tool()
        self.register_only_image_tools()
        self.add_separator_tool()
        self.register_other_tools()
        self.get_default_tool().activate()

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

class BaseCurvePlotWidget(QSplitter):
    """
    Construct a BaseCurvePlotWidget object, which includes:
        * A plot (:py:class:`guiqwt.curve.CurvePlot`)
        * An `item list` panel (:py:class:`guiqwt.curve.PlotItemList`)
        
    This object does nothing in itself because plot and panels are not 
    connected to each other.
    See children class :py:class:`guiqwt.plot.CurvePlotWidget`
    """
    def __init__(self, parent=None, title=None, xlabel=None, ylabel=None,
                 section="plot", show_itemlist=False, gridparam=None):
        QSplitter.__init__(self, Qt.Horizontal, parent)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.plot = CurvePlot(parent=self,
                              title=title, xlabel=xlabel, ylabel=ylabel,
                              section=section, gridparam=gridparam)
        self.addWidget(self.plot)
        self.itemlist = PlotItemList(self)
        self.itemlist.setVisible(show_itemlist)
        self.addWidget(self.itemlist)
        configure_plot_splitter(self)

class CurvePlotWidget(BaseCurvePlotWidget, PlotManager):
    """
    Construct a CurvePlotWidget object: plotting widget with integrated 
    plot manager
        * parent: parent widget
        * title: plot title
        * xlabel: (bottom axis title, top axis title) or bottom axis title only
        * ylabel: (left axis title, right axis title) or left axis title only
    """
    def __init__(self, parent=None, title=None, xlabel=None, ylabel=None,
                 section="plot", show_itemlist=False, gridparam=None):
        BaseCurvePlotWidget.__init__(self, parent, title, xlabel, ylabel,
                                     section, show_itemlist, gridparam)
        PlotManager.__init__(self, main=self)
        
        # Configuring plot manager
        self.add_plot(self.plot)
        self.add_panel(self.itemlist)
        
class CurvePlotDialog(QDialog, PlotManager):
    """
    Construct a CurvePlotDialog object: plotting dialog box with integrated 
    plot manager
        * wintitle: window title
        * icon: window icon
        * edit: editable state
        * toolbar: show/hide toolbar
        * options: options sent to the :py:class:`guiqwt.curve.CurvePlot` object
          (dictionary)
        * parent: parent widget
    """
    def __init__(self, wintitle="guiqwt plot", icon="guiqwt.png",
                 edit=False, toolbar=False, options=None, parent=None):
        QDialog.__init__(self, parent)
        PlotManager.__init__(self, main=self)

        self.edit = edit
        self.setWindowTitle(wintitle)
        if isinstance(icon, basestring):
            icon = get_icon(icon)
        self.setWindowIcon(icon)
        self.setMinimumSize(320, 240)
        self.resize(640, 480)
        self.setWindowFlags(Qt.Window)
        
        self.plot_layout = QGridLayout()
        
        if options is None:
            options = {}
            
        self.create_plot(options)
        
        self.vlayout = QVBoxLayout(self)
        
        self.toolbar = QToolBar(_("Tools"))
        if not toolbar:
            self.toolbar.hide()
        self.vlayout.addWidget(self.toolbar)
        
        self.setLayout(self.vlayout)
        self.vlayout.addLayout(self.plot_layout)
        
        if self.edit:
            self.button_layout = QHBoxLayout()
            self.install_button_layout()
            self.vlayout.addLayout(self.button_layout)
        
        # Configuring plot manager
        self.add_toolbar(self.toolbar, "default")
        self.register_tools()
        
    def install_button_layout(self):
        """
        Install standard buttons (OK, Cancel) in dialog button box layout 
        (:py:attr:`guiqwt.plot.CurvePlotDialog.button_layout`)
        
        This method may be overriden to customize the button box
        """
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.connect(bbox, SIGNAL("accepted()"), SLOT("accept()"))
        self.connect(bbox, SIGNAL("rejected()"), SLOT("reject()"))
        self.button_layout.addWidget(bbox)

    def register_tools(self):
        """Derived classes can override this method
        to provide a fully customized set of tools"""
        self.register_curve_tools()

    def create_plot(self, options):
        """
        Create the plotting widget (which is an instance of class 
        :py:class:`guiqwt.plot.BaseCurvePlotWidget`), add it to the dialog box 
        main layout (:py:attr:`guiqwt.plot.CurvePlotDialog.plot_layout`) and 
        then add the `item list` panel

        May be overriden to customize the plot layout 
        (:py:attr:`guiqwt.plot.CurvePlotDialog.plot_layout`)
        """
        plotwidget = BaseCurvePlotWidget(self, **options)
        self.plot_layout.addWidget(plotwidget, 0, 0)
        
        # Configuring plot manager
        self.add_plot(plotwidget.plot)
        self.add_panel(plotwidget.itemlist)


#===============================================================================
# Image Plot Widget/Dialog with integrated Levels Histogram and other widgets
#===============================================================================
class BaseImagePlotWidget(QSplitter):
    """
    Construct a BaseImagePlotWidget object, which includes:
        * A plot (:py:class:`guiqwt.curve.CurvePlot`)
        * An `item list` panel (:py:class:`guiqwt.curve.PlotItemList`)
        * A `contrast adjustment` panel 
          (:py:class:`guiqwt.histogram.ContrastAdjustment`)
        * An `X-axis cross section` panel
          (:py:class:`guiqwt.histogram.XCrossSection`)
        * An `Y-axis cross section` panel
          (:py:class:`guiqwt.histogram.YCrossSection`)
        
    This object does nothing in itself because plot and panels are not 
    connected to each other.
    See children class :py:class:`guiqwt.plot.ImagePlotWidget`
    """
    def __init__(self, parent=None, title="",
                 xlabel=("", ""), ylabel=("", ""), zlabel=None, yreverse=True,
                 colormap="jet", aspect_ratio=1.0, lock_aspect_ratio=True,
                 show_contrast=False, show_itemlist=False, show_xsection=False,
                 show_ysection=False, xsection_pos="top", ysection_pos="right",
                 gridparam=None):
        QSplitter.__init__(self, Qt.Vertical, parent)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.sub_splitter = QSplitter(Qt.Horizontal, self)
        self.plot = ImagePlot(parent=self, title=title,
                              xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                              yreverse=yreverse, aspect_ratio=aspect_ratio,
                              lock_aspect_ratio=lock_aspect_ratio,
                              gridparam=gridparam)

        from guiqwt.cross_section import YCrossSection
        self.ycsw = YCrossSection(self, position=ysection_pos)
        self.ycsw.setVisible(show_ysection)
        
        from guiqwt.cross_section import XCrossSection
        self.xcsw = XCrossSection(self)
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

class ImagePlotWidget(BaseImagePlotWidget, PlotManager):
    """
    Construct a ImagePlotWidget object: plotting widget with integrated 
    plot manager
        * parent: parent widget
        * title: plot title (string)
        * xlabel, ylabel, zlabel: resp. bottom, left and right axis titles 
          (strings)
        * yreverse: reversing Y-axis (bool)
        * aspect_ratio: height to width ratio (float)
        * lock_aspect_ratio: locking aspect ratio (bool)
        * show_contrast: showing contrast adjustment tool (bool)
        * show_xsection: showing x-axis cross section plot (bool)
        * show_ysection: showing y-axis cross section plot (bool)
        * xsection_pos: x-axis cross section plot position 
          (string: "top", "bottom")
        * ysection_pos: y-axis cross section plot position 
          (string: "left", "right")
    """
    def __init__(self, parent=None, title="",
                 xlabel=("", ""), ylabel=("", ""), zlabel=None, yreverse=True,
                 colormap="jet", aspect_ratio=1.0, lock_aspect_ratio=True,
                 show_contrast=False, show_itemlist=False, show_xsection=False,
                 show_ysection=False, xsection_pos="top", ysection_pos="right",
                 gridparam=None):
        BaseImagePlotWidget.__init__(self, parent, title, xlabel, ylabel,
                 zlabel, yreverse, colormap, aspect_ratio, lock_aspect_ratio,
                 show_contrast, show_itemlist, show_xsection, show_ysection,
                 xsection_pos, ysection_pos, gridparam)
        PlotManager.__init__(self, main=self)
        
        # Configuring plot manager
        self.add_plot(self.plot)
        self.add_panel(self.itemlist)
        self.add_panel(self.xcsw)
        self.add_panel(self.ycsw)
        self.add_panel(self.contrast)

class ImagePlotDialog(CurvePlotDialog):
    """
    Construct a ImagePlotDialog object: plotting dialog box with integrated 
    plot manager
        * wintitle: window title
        * icon: window icon
        * edit: editable state
        * toolbar: show/hide toolbar
        * options: options sent to the :py:class:`guiqwt.image.ImagePlot` object
          (dictionary)
        * parent: parent widget
    """
    def __init__(self, wintitle="guiqwt imshow", icon="guiqwt.png",
                 edit=False, toolbar=False, options=None, parent=None):
        CurvePlotDialog.__init__(self, wintitle=wintitle, icon=icon, edit=edit,
                                 toolbar=toolbar, options=options,
                                 parent=parent)

    def create_plot(self, options, row=0, column=0, rowspan=1, columnspan=1):
        """
        Create the plotting widget (which is an instance of class 
        :py:class:`guiqwt.plot.BaseImagePlotWidget`), add it to the dialog box 
        main layout (:py:attr:`guiqwt.plot.CurvePlotDialog.plot_layout`) and 
        then add the `item list`, `contrast adjustment` and X/Y axes 
        cross section panels.

        May be overriden to customize the plot layout 
        (:py:attr:`guiqwt.plot.CurvePlotDialog.plot_layout`)
        """
        plotwidget = BaseImagePlotWidget(self, **options)
        self.plot_layout.addWidget(plotwidget, row, column, rowspan, columnspan)
        
        # Configuring plot manager
        self.add_plot(plotwidget.plot)
        self.add_panel(plotwidget.itemlist)
        self.add_panel(plotwidget.xcsw)
        self.add_panel(plotwidget.ycsw)
        self.add_panel(plotwidget.contrast)
