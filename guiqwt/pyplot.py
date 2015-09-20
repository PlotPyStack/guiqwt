# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

# pylint: disable=C0103

"""
guiqwt.pyplot
-------------

The `pyplot` module provides an interactive plotting interface similar to 
`Matplotlib`'s, i.e. with MATLAB-like syntax.

The :py:mod:`guiqwt.pyplot` module was designed to be as close as possible 
to the :py:mod:`matplotlib.pyplot` module, so that one could easily switch 
between these two modules by simply changing the import statement. Basically,
if `guiqwt` does support the plotting commands called in your script, replacing 
``import matplotlib.pyplot`` by ``import guiqwt.pyplot`` should suffice, as 
shown in the following example:
    
    * Simple example using `matplotlib`::
    
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.linspace(-10, 10)
        plt.plot(x, x**2, 'r+')
        plt.show()

    * Switching from `matplotlib` to `guiqwt` is trivial::
    
        import guiqwt.pyplot as plt # only this line has changed!
        import numpy as np
        x = np.linspace(-10, 10)
        plt.plot(x, x**2, 'r+')
        plt.show()

Examples
~~~~~~~~

>>> import numpy as np
>>> from guiqwt.pyplot import * # ugly but acceptable in an interactive session
>>> ion() # switching to interactive mode
>>> x = np.linspace(-5, 5, 1000)
>>> figure(1)
>>> subplot(2, 1, 1)
>>> plot(x, np.sin(x), "r+")
>>> plot(x, np.cos(x), "g-")
>>> errorbar(x, -1+x**2/20+.2*np.random.rand(len(x)), x/20)
>>> xlabel("Axe x")
>>> ylabel("Axe y")
>>> subplot(2, 1, 2)
>>> img = np.fromfunction(lambda x, y: np.sin((x/200.)*(y/200.)**2), (1000, 1000))
>>> xlabel("pixels")
>>> ylabel("pixels")
>>> zlabel("intensity")
>>> gray()
>>> imshow(img)
>>> figure("plotyy")
>>> plotyy(x, np.sin(x), x, np.cos(x))
>>> ylabel("sinus", "cosinus")
>>> show()

Reference
~~~~~~~~~

.. autofunction:: interactive
.. autofunction:: ion
.. autofunction:: ioff

.. autofunction:: figure
.. autofunction:: gcf
.. autofunction:: gca
.. autofunction:: show
.. autofunction:: subplot
.. autofunction:: close

.. autofunction:: title
.. autofunction:: xlabel
.. autofunction:: ylabel
.. autofunction:: zlabel

.. autofunction:: yreverse
.. autofunction:: grid
.. autofunction:: legend
.. autofunction:: colormap

.. autofunction:: savefig

.. autofunction:: plot
.. autofunction:: plotyy
.. autofunction:: semilogx
.. autofunction:: semilogy
.. autofunction:: loglog
.. autofunction:: errorbar
.. autofunction:: hist
.. autofunction:: imshow
.. autofunction:: pcolor
"""

from __future__ import print_function

import sys
from guidata.qt.QtGui import (QMainWindow, QPrinter, QPainter, QFrame,
                              QVBoxLayout, QGridLayout, QToolBar, QPixmap,
                              QImageWriter)
from guidata.qt.QtCore import QRect, Qt, QBuffer, QIODevice
from guidata.qt import PYQT5

import guidata
from guidata.configtools import get_icon
from guidata.py3compat import is_text_string, to_text_string

# Local imports
from guiqwt.config import _
from guiqwt.plot import PlotManager
from guiqwt.image import ImagePlot, INTERP_NEAREST, INTERP_LINEAR, INTERP_AA
from guiqwt.curve import CurvePlot, PlotItemList
from guiqwt.histogram import ContrastAdjustment
from guiqwt.cross_section import XCrossSection, YCrossSection
from guiqwt.builder import make


_interactive = False
_figures = {}
_current_fig = None
_current_axes = None


class Window(QMainWindow):
    def __init__(self, wintitle):
        super(Window, self).__init__()
        self.default_tool = None
        self.plots = []
        self.itemlist = PlotItemList(None)
        self.contrast = ContrastAdjustment(None)
        self.xcsw = XCrossSection(None)
        self.ycsw = YCrossSection(None)
        
        self.manager = PlotManager(self)
        self.toolbar = QToolBar(_("Tools"), self)
        self.manager.add_toolbar(self.toolbar, "default")
        self.toolbar.setMovable(True)
        self.toolbar.setFloatable(True)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        frame = QFrame(self)
        self.setCentralWidget(frame)
        self.layout = QGridLayout()
        layout = QVBoxLayout(frame)
        frame.setLayout(layout)
        layout.addLayout(self.layout)
        self.frame = frame

        self.setWindowTitle(wintitle)
        self.setWindowIcon(get_icon('guiqwt.svg'))

    def closeEvent(self, event):
        global _figures, _current_fig, _current_axes
        figure_title = to_text_string(self.windowTitle())
        if _figures.pop(figure_title) == _current_fig:
            _current_fig = None
            _current_axes = None
        self.itemlist.close()
        self.contrast.close()
        self.xcsw.close()
        self.ycsw.close()
        event.accept()
        
    def add_plot(self, i, j, plot):
        self.layout.addWidget(plot, i, j)
        self.manager.add_plot(plot)
        self.plots.append(plot)

    def replot(self):
        for plot in self.plots:
            plot.replot()
            item = plot.get_default_item()
            if item is not None:
                plot.set_active_item(item)
                item.unselect()
            
    def add_panels(self, images=False):
        self.manager.add_panel(self.itemlist)
        if images:
            for panel in (self.ycsw, self.xcsw, self.contrast):
                panel.hide()
                self.manager.add_panel(panel)
            
    def register_tools(self, images=False):
        if images:
            self.manager.register_all_image_tools()
        else:
            self.manager.register_all_curve_tools()
    
    def display(self):
        self.show()
        self.replot()
        self.manager.get_default_tool().activate()
        self.manager.update_tools_status()


class Figure(object):
    def __init__(self, title):
        self.axes = {}
        self.title = title
        self.win = None
        self.app = None

    def get_axes(self, i, j):
        if (i, j) in self.axes:
            return self.axes[(i, j)]

        ax = Axes()
        self.axes[(i, j)] = ax
        return ax

    def build_window(self):
        self.app = guidata.qapplication()
        self.win = Window(wintitle=self.title)
        images = False
        for (i, j), ax in list(self.axes.items()):
            ax.setup_window(i, j, self.win)
            if ax.images:
                images = True
        self.win.add_panels(images=images)
        self.win.register_tools(images=images)

    def show(self):
        if not self.win:
            self.build_window()
        self.win.display()
        
    def save(self, fname, format, draft):
        if is_text_string(fname):
            if format == "pdf":
                self.app = guidata.qapplication()
                if draft:
                    mode = QPrinter.ScreenResolution
                else:
                    mode = QPrinter.HighResolution
                printer = QPrinter(mode)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOrientation(QPrinter.Landscape)
                printer.setOutputFileName(fname)
                printer.setCreator('guiqwt.pyplot')
                self.print_(printer)
            else:
                if self.win is None:
                    self.show()
                if PYQT5:
                    pixmap = self.win.centralWidget().grab()
                else:
                    pixmap = QPixmap.grabWidget(self.win.centralWidget())
                pixmap.save(fname, format.upper())
        else:
            # Buffer
            fd = fname
            assert hasattr(fd, 'write'), "object is not file-like as expected"
            if self.win is None:
                self.show()
            pixmap = QPixmap.grabWidget(self.win.centralWidget())
            buff = QBuffer()
            buff.open(QIODevice.ReadWrite)
            pixmap.save(buff, format.upper())
            fd.write(buff.data())
            buff.close()
            fd.seek(0)
        
    def print_(self, device):
        if not self.win:
            self.build_window()
        W = device.width()
        H = device.height()
        from numpy import array
        coords = array(list(self.axes.keys()))
        imin = coords[:, 0].min()
        imax = coords[:, 0].max()
        jmin = coords[:, 1].min()
        jmax = coords[:, 1].max()
        w = W/(jmax-jmin+1)
        h = H/(imax-imin+1)
        paint = QPainter(device)
        for (i, j), ax in list(self.axes.items()):
            oy = (i-imin)*h
            ox = (j-jmin)*w
            ax.widget.print_(paint, QRect(ox, oy, w, h))
        

def do_mainloop(mainloop):
    global _current_fig
    if not _current_fig:
        print("Warning: must create a figure before showing it", file=sys.stderr)
    elif mainloop:
        app = guidata.qapplication()
        app.exec_()
        

class Axes(object):
    def __init__(self):
        self.plots = []
        self.images = []
        self.last = None
        self.legend_position = None
        self.grid = False
        self.xlabel = ("", "")
        self.ylabel = ("", "")
        self.xcolor = ("black", "black") # axis label colors
        self.ycolor = ("black", "black") # axis label colors
        self.zlabel = None
        self.yreverse = False
        self.colormap = "jet"
        self.xscale = 'lin'
        self.yscale = 'lin'
        self.xlimits = None
        self.ylimits = None
        self.widget = None
        self.main_widget = None

    def add_legend(self, position):
        self.legend_position = position

    def set_grid(self, grid):
        self.grid = grid
        
    def set_xlim(self, xmin, xmax):
        self.xlimits = xmin, xmax
        self._update_plotwidget()
        
    def set_ylim(self, ymin, ymax):
        self.ylimits = ymin, ymax
        self._update_plotwidget()

    def add_plot(self, item):
        self.plots.append(item)
        self.last = item

    def add_image(self, item):
        self.images.append(item)
        self.last = item

    def setup_window(self, i, j, win):
        if self.images:
            plot = self.setup_image(i, j, win)
        else:
            plot = self.setup_plot(i, j, win)
        self.widget = plot
        plot.do_autoscale()
        self._update_plotwidget()
        
    def _update_plotwidget(self):
        p = self.main_widget
        if p is None:
            return
        if self.grid:
            p.gridparam.maj_xenabled = True
            p.gridparam.maj_yenabled = True
            p.gridparam.update_grid(p)
        p.set_axis_color('bottom', self.xcolor[0])
        p.set_axis_color('top', self.xcolor[1])
        p.set_axis_color('left', self.ycolor[0])
        p.set_axis_color('right', self.ycolor[1])
        if self.xlimits is not None:
            p.set_axis_limits('bottom', *self.xlimits)
        if self.ylimits is not None:
            p.set_axis_limits('left', *self.ylimits)

    def setup_image(self, i, j, win):
        p = ImagePlot(win, xlabel=self.xlabel, ylabel=self.ylabel,
                      zlabel=self.zlabel, yreverse=self.yreverse)
        self.main_widget = p
        win.add_plot(i, j, p)
        for item in self.images+self.plots:
            if item in self.images:
                item.set_color_map(self.colormap)
            p.add_item(item)
        if self.legend_position is not None:
            p.add_item(make.legend(self.legend_position))
        return p

    def setup_plot(self, i, j, win):
        p = CurvePlot(win, xlabel=self.xlabel, ylabel=self.ylabel)
        self.main_widget = p
        win.add_plot(i, j, p)
        for item in self.plots:
            p.add_item(item)
        p.enable_used_axes()
        active_item = p.get_active_item(force=True)
        p.set_scales(self.xscale, self.yscale)
        active_item.unselect()
        if self.legend_position is not None:
            p.add_item(make.legend(self.legend_position))
        return p
            

def _make_figure_title(N=None):
    global _figures
    if N is None:
        N = len(_figures)+1
    if is_text_string(N):
        return N
    else:
        return "Figure %d" % N

def figure(N=None):
    """Create a new figure"""
    global _figures, _current_fig, _current_axes
    title = _make_figure_title(N)
    if title in _figures:
        f = _figures[title]
    else:
        f = Figure(title)
        _figures[title] = f
    _current_fig = f
    _current_axes = None
    return f

def gcf():
    """Get current figure"""
    global _current_fig
    if _current_fig:
        return _current_fig
    else:
        return figure()

def gca():
    """Get current axes"""
    global _current_axes
    if not _current_axes:
        axes = gcf().get_axes(1, 1)
        _current_axes = axes
    return _current_axes
 
def show(mainloop=True):
    """
    Show all figures and enter Qt event loop    
    This should be the last line of your script
    """
    global _figures, _interactive
    for fig in list(_figures.values()):
        fig.show()
    if not _interactive:
        do_mainloop(mainloop)

def _show_if_interactive():
    global _interactive
    if _interactive:
        show()


def subplot(n, m, k):
    """
    Create a subplot command
    
    Example::

        import numpy as np
        x = np.linspace(-5, 5, 1000)
        figure(1)
        subplot(2, 1, 1)
        plot(x, np.sin(x), "r+")
        subplot(2, 1, 2)
        plot(x, np.cos(x), "g-")
        show()
    """
    global _current_axes
    lig = (k-1)/m
    col = (k-1)%m
    fig = gcf()
    axe = fig.get_axes(lig, col)
    _current_axes = axe
    return axe

def plot(*args, **kwargs):
    """
    Plot curves
    
    Example::
    
        import numpy as np
        x = np.linspace(-5, 5, 1000)
        plot(x, np.sin(x), "r+")
        plot(x, np.cos(x), "g-")
        show()
    """
    axe = gca()
    curves = make.mcurve(*args, **kwargs)
    if not isinstance(curves, list):
        curves = [curves]
    for curve in curves:
        axe.add_plot(curve)
    _show_if_interactive()
    return curves

def plotyy(x1, y1, x2, y2):
    """
    Plot curves with two different y axes
    
    Example::
        
        import numpy as np
        x = np.linspace(-5, 5, 1000)
        plotyy(x, np.sin(x), x, np.cos(x))
        ylabel("sinus", "cosinus")
        show()
    """
    axe = gca()
    curve1 = make.mcurve(x1, y1, yaxis='left')
    curve2 = make.mcurve(x2, y2, yaxis='right')
    axe.ycolor = (curve1.curveparam.line.color, curve2.curveparam.line.color)
    axe.add_plot(curve1)
    axe.add_plot(curve2)
    _show_if_interactive()
    return [curve1, curve2]

def hist(data, bins=None, logscale=None, title=None, color=None):
    """
    Plot 1-D histogram
    
    Example::
        
        from numpy.random import normal
        data = normal(0, 1, (2000, ))
        hist(data)
        show()
    """
    axe = gca()
    curve = make.histogram(data, bins=bins, logscale=logscale,
                           title=title, color=color, yaxis='left')
    axe.add_plot(curve)
    _show_if_interactive()
    return [curve]

def semilogx(*args, **kwargs):
    """
    Plot curves with logarithmic x-axis scale
    
    Example::
        
        import numpy as np
        x = np.linspace(-5, 5, 1000)
        semilogx(x, np.sin(12*x), "g-")
        show()
    """
    axe = gca()
    axe.xscale = 'log'
    curve = make.mcurve(*args, **kwargs)
    axe.add_plot(curve)
    _show_if_interactive()
    return [curve]
    
def semilogy(*args, **kwargs):
    """
    Plot curves with logarithmic y-axis scale
    
    Example::
        
        import numpy as np
        x = np.linspace(-5, 5, 1000)
        semilogy(x, np.sin(12*x), "g-")
        show()
    """
    axe = gca()
    axe.yscale = 'log'
    curve = make.mcurve(*args, **kwargs)
    axe.add_plot(curve)
    _show_if_interactive()
    return [curve]
    
def loglog(*args, **kwargs):
    """
    Plot curves with logarithmic x-axis and y-axis scales
    
    Example::
        
        import numpy as np
        x = np.linspace(-5, 5, 1000)
        loglog(x, np.sin(12*x), "g-")
        show()
    """
    axe = gca()
    axe.xscale = 'log'
    axe.yscale = 'log'
    curve = make.mcurve(*args, **kwargs)
    axe.add_plot(curve)
    _show_if_interactive()
    return [curve]

def errorbar(*args, **kwargs):
    """
    Plot curves with error bars
    
    Example::
        
        import numpy as np
        x = np.linspace(-5, 5, 1000)
        errorbar(x, -1+x**2/20+.2*np.random.rand(len(x)), x/20)
        show()
    """
    axe = gca()
    curve = make.merror(*args, **kwargs)
    axe.add_plot(curve)
    _show_if_interactive()
    return [curve]

def imread(fname, to_grayscale=False):
    """Read data from *fname*"""
    from guiqwt import io
    return io.imread(fname, to_grayscale=to_grayscale)

def imshow(data, interpolation=None, mask=None):
    """
    Display the image in *data* to current axes
    interpolation: 'nearest', 'linear' (default), 'antialiasing'
    
    Example::
        
        import numpy as np
        x = np.linspace(-5, 5, 1000)
        img = np.fromfunction(lambda x, y: np.sin((x/200.)*(y/200.)**2), (1000, 1000))
        gray()
        imshow(img)
        show()
    """
    axe = gca()
    import numpy as np
    if isinstance(data, np.ma.MaskedArray) and mask is None:
        mask = data.mask
        data = data.data
    if mask is None:
        img = make.image(data)
    else:
        img = make.maskedimage(data, mask, show_mask=True)
    if interpolation is not None:
        interp_dict = {'nearest': INTERP_NEAREST,
                       'linear': INTERP_LINEAR,
                       'antialiasing': INTERP_AA}
        assert interpolation in interp_dict, "invalid interpolation option"
        img.set_interpolation(interp_dict[interpolation], size=5)
    axe.add_image(img)
    axe.yreverse = True
    _show_if_interactive()
    return [img]

def pcolor(*args):
    """
    Create a pseudocolor plot of a 2-D array
    
    Example::
    
        import numpy as np
        r = np.linspace(1., 16, 100)
        th = np.linspace(0., np.pi, 100)
        R, TH = np.meshgrid(r, th)
        X = R*np.cos(TH)
        Y = R*np.sin(TH)
        Z = 4*TH+R
        pcolor(X, Y, Z)
        show()
    """
    axe = gca()
    img = make.pcolor(*args)
    axe.add_image(img)
    axe.yreverse = len(args) == 1
    _show_if_interactive()
    return [img]

def interactive(state):
    """Toggle interactive mode"""
    global _interactive
    _interactive = state

def ion():
    """Turn interactive mode on"""
    interactive(True)

def ioff():
    """Turn interactive mode off"""
    interactive(False)

#TODO: The following functions (title, xlabel, ...) should update an already 
#      shown figure to be compatible with interactive mode -- for now it just 
#      works if these functions are called before showing the figure
def title(text):
    """Set current figure title"""
    global _figures
    fig = gcf()
    _figures.pop(fig.title)
    fig.title = text
    _figures[text] = fig

def xlabel(bottom="", top=""):
    """Set current x-axis label"""
    assert is_text_string(bottom) and is_text_string(top)
    axe = gca()
    axe.xlabel = (bottom, top)

def ylabel(left="", right=""):
    """Set current y-axis label"""
    assert is_text_string(left) and is_text_string(right)
    axe = gca()
    axe.ylabel = (left, right)

def zlabel(label):
    """Set current z-axis label"""
    assert is_text_string(label)
    axe = gca()
    axe.zlabel = label
    
def yreverse(reverse):
    """
    Set y-axis direction of increasing values
    
    reverse = False (default)
        y-axis values increase from bottom to top
    
    reverse = True
        y-axis values increase from top to bottom
    """
    assert isinstance(reverse, bool)
    axe = gca()
    axe.yreverse = reverse

def grid(act):
    """Toggle grid visibility"""
    axe = gca()
    axe.set_grid(act)

def legend(pos="TR"):
    """Add legend to current axes (pos='TR', 'TL', 'BR', ...)"""
    axe = gca()
    axe.add_legend(pos)

def colormap(name):
    """Set color map to *name*"""
    axe = gca()
    axe.colormap = name

def _add_colormaps(glbs):
    from guiqwt.colormap import get_colormap_list
    for cmap_name in get_colormap_list():
        glbs[cmap_name] = lambda name=cmap_name: colormap(name)
        glbs[cmap_name].__doc__ = "Set color map to '%s'" % cmap_name
_add_colormaps(globals())

def close(N=None, all=False):
    """Close figure"""
    global _figures, _current_fig, _current_axes
    if all:
        _figures = {}
        _current_fig = None
        _current_axes = None
        return
    if N is None:
        fig = gcf()
    else:
        fig = figure(N)
    fig.close()

def savefig(fname, format=None, draft=False):
    """
    Save figure
    
    Currently supports PDF and PNG formats only
    """
    if not is_text_string(fname) and format is None:
        # Buffer/fd
        format = 'png'
    if format is None:
        format = fname.rsplit(".", 1)[-1].lower()
        fmts = [str(fmt).lower()
                for fmt in QImageWriter.supportedImageFormats()]
        assert format in fmts, _("Function 'savefig' currently supports the "
                                 "following formats:\n%s"
                                 ) % ','.join(fmts)
    else:
        format = format.lower()
    fig = gcf()
    fig.save(fname, format, draft)
