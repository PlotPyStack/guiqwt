# -*- coding: utf-8 -*-
#
# Copyright © 2010-2011 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
SIFT, the Signal and Image Filtering Tool
Simple signal and image processing application based on guiqwt and guidata
"""

SHOW = True # Show test in GUI-based test launcher

from PyQt4.QtGui import (QMainWindow, QMessageBox, QSplitter, QListWidget,
                         QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
                         QTabWidget)
from PyQt4.QtCore import Qt, QT_VERSION_STR, PYQT_VERSION_STR, SIGNAL

import sys, platform, os.path as osp
import numpy as np

from guidata.dataset.datatypes import DataSet
from guidata.dataset.dataitems import (IntItem, FloatArrayItem, StringItem,
                                       ChoiceItem, FloatItem)
from guidata.dataset.qtwidgets import DataSetEditGroupBox
from guidata.configtools import get_icon
from guidata.qthelpers import create_action, add_actions, get_std_icon
from guidata.qtwidgets import DockableWidget
from guidata.utils import update_dataset

from guiqwt.config import _
from guiqwt.plot import CurveWidget, ImageWidget
from guiqwt.builder import make

APP_NAME = _("Sift")
APP_DESC = _("""Signal and Image Filtering Tool<br>
Simple signal and image processing application based on guiqwt and guidata""")
VERSION = '0.2.0'


class SignalParam(DataSet):
    title = StringItem(_("Title"), default=_("Untitled"))
    xydata = FloatArrayItem(_("Data"), transpose=True, minmax="rows")
    def copy_data_from(self, other):
        self.xydata = np.array(other.xydata, copy=True)
    def get_data(self):
        if self.xydata is not None:
            return self.xydata[1]
    def set_data(self, data):
        self.xydata[1] = data
    data = property(get_data, set_data)

class SignalParamNew(DataSet):
    title = StringItem(_("Title"), default=_("Untitled"))
    xmin = FloatItem("Xmin", default=-10.)
    xmax = FloatItem("Xmax", default=10.)
    size = IntItem(_("Size"), help=_("Signal size (total number of points)"),
                   min=1, default=500)
    type = ChoiceItem(_("Type"),
                      (("rand", _("random")), ("zeros", _("zeros")),
                       ("gauss", _("gaussian"))))


class ImageParam(DataSet):
    title = StringItem(_("Title"), default=_("Untitled"))
    data = FloatArrayItem(_("Data"))
    def copy_data_from(self, other):
        self.data = np.array(other.data, copy=True)

class ImageParamNew(DataSet):
    title = StringItem(_("Title"), default=_("Untitled"))
    height = IntItem(_("Height"), help=_("Image height (total number of rows)"),
                     min=1, default=500)
    width = IntItem(_("Width"), help=_("Image width (total number of columns)"),
                    min=1, default=500)
    dtype = ChoiceItem(_("Data type"),
                       ((np.uint8, "uint8"), (np.int16, "uint16"),
                        (np.float32, "float32"), (np.float64, "float64"),
                        ))
    type = ChoiceItem(_("Type"),
                      (("zeros", _("zeros")), ("empty", _("empty")),
                       ("rand", _("random")),
                        ))


class ObjectFT(QSplitter):
    """Object handling the item list, the selected item properties and plot"""
    PARAMCLASS = None
    PREFIX = None
    def __init__(self, parent, plot):
        super(ObjectFT, self).__init__(Qt.Vertical, parent)
        self.plot = plot
        self.objects = [] # signals or images
        self.items = [] # associated plot items
        self.listwidget = None
        self.properties = None
        self._hsplitter = None
        
        self.file_actions = None
        self.edit_actions = None
        self.operation_actions = None
        self.processing_actions = None
        
        self.number = 0
        
        self.directory = "" # last browsed directory

        # Object selection dependent actions
        self.actlist_1more = []
        self.actlist_2more = []
        self.actlist_1 = []
        self.actlist_2 = []
        
    #------Setup widget, menus, actions
    def setup(self, toolbar):
        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(QListWidget.ExtendedSelection)
        self.properties = DataSetEditGroupBox(_("Properties"), self.PARAMCLASS)
        self.properties.setEnabled(False)

        self.connect(self.listwidget, SIGNAL("currentRowChanged(int)"),
                     self.current_item_changed)
        self.connect(self.listwidget, SIGNAL("itemSelectionChanged()"),
                     self.selection_changed)
        self.connect(self.properties, SIGNAL("apply_button_clicked()"),
                     self.properties_changed)
        
        properties_stretched = QWidget()
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.properties)
        hlayout.addStretch()
        vlayout = QVBoxLayout()
        vlayout.addLayout(hlayout)
        vlayout.addStretch()
        properties_stretched.setLayout(vlayout)
        
        self.addWidget(self.listwidget)
        self.addWidget(properties_stretched)

        # Edit actions
        duplicate_action = create_action(self, _("Duplicate"),
                                         icon=get_icon('copy.png'),
                                         triggered=self.duplicate_object)
        self.actlist_1 += [duplicate_action]
        remove_action = create_action(self, _("Remove"),
                                      icon=get_icon('delete.png'),
                                      triggered=self.remove_object)
        self.actlist_1more += [remove_action]
        self.edit_actions = [duplicate_action, remove_action]
        
        # Operation actions
        sum_action = create_action(self, _("Sum"), triggered=self.compute_sum)
        diff_action = create_action(self, _("Difference"),
                                    triggered=self.compute_difference)
        prod_action = create_action(self, _("Product"),
                                    triggered=self.compute_product)
        div_action = create_action(self, _("Division"),
                                   triggered=self.compute_division)
        self.actlist_2more += [sum_action, prod_action]
        self.actlist_2 += [diff_action, div_action]
        self.operation_actions = [sum_action, diff_action, prod_action,
                                  div_action]

    #------GUI refresh/setup
    def current_item_changed(self, row):
        if row != -1:
            update_dataset(self.properties.dataset, self.objects[row])
            self.properties.get()

    def _get_selected_rows(self):
        return [index.row() for index in
                self.listwidget.selectionModel().selectedRows()]
        
    def selection_changed(self):
        """Signal list: selection changed"""
        row = self.listwidget.currentRow()
        self.properties.setDisabled(row == -1)
        self.refresh_plot()
        nbrows = len(self._get_selected_rows())
        for act in self.actlist_1more:
            act.setEnabled(nbrows >= 1)
        for act in self.actlist_2more:
            act.setEnabled(nbrows >= 2)
        for act in self.actlist_1:
            act.setEnabled(nbrows == 1)
        for act in self.actlist_2:
            act.setEnabled(nbrows == 2)
            
    def make_item(self, row):
        raise NotImplementedError
        
    def update_item(self, row):
        raise NotImplementedError
        
    def refresh_plot(self):
        for item in self.items:
            if item is not None:
                item.hide()
        for row in self._get_selected_rows():
            item = self.items[row]
            if item is None:
                item = self.make_item(row)
                self.plot.add_item(item)
            else:
                self.update_item(row)
                self.plot.set_item_visible(item, True)
        self.plot.do_autoscale()
        
    def refresh_list(self):
        self.listwidget.clear()
        self.listwidget.addItems(["%s%03d: %s" % (self.PREFIX, i, obj.title)
                                  for i, obj in enumerate(self.objects)])
        
    def properties_changed(self):
        """The properties 'Apply' button was clicked: updating signal"""
        row = self.listwidget.currentRow()
        update_dataset(self.objects[row], self.properties.dataset)
        self.refresh_list()
        self.listwidget.setCurrentRow(row)
        self.refresh_plot()
    
    def add_object(self, obj):
        self.objects.append(obj)
        self.items.append(None)
        self.refresh_list()
        self.listwidget.setCurrentRow(len(self.objects)-1)
        self.emit(SIGNAL('object_added()'))
        
    #------Edit operations
    def duplicate_object(self):
        row = self._get_selected_rows()[0]
        obj = self.objects[row]
        objcopy = self.PARAMCLASS()
        objcopy.title = obj.title
        objcopy.copy_data_from(obj)
        self.objects.insert(row+1, objcopy)
        self.items.insert(row+1, None)
        self.refresh_list()
        self.listwidget.setCurrentRow(row+1)
        self.refresh_plot()
    
    def remove_object(self):
        rows = sorted(self._get_selected_rows(), reverse=True)
        for row in rows:
            self.objects.pop(row)
            item = self.items.pop(row)
            self.plot.del_item(item)
        self.refresh_list()
        self.refresh_plot()
        
    #------Operations
    def apply_sum_func(self, sumobj, obj):
        raise NotImplementedError
        
    def apply_diff_func(self, diffobj, obj0, obj1):
        raise NotImplementedError
        
    def compute_sum(self):
        rows = self._get_selected_rows()
        sumobj = self.PARAMCLASS()
        sumobj.title = "+".join(["%s%03d" % (self.PREFIX, row) for row in rows])
        try:
            for row in rows:
                obj = self.objects[row]
                if sumobj.data is None:
                    sumobj.copy_data_from(obj)
                else:
                    sumobj.data += obj.data
        except Exception, msg:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent(), APP_NAME,
                                 _(u"Error:")+"\n%s" % str(msg))
            return
        self.add_object(sumobj)
    
    def compute_product(self):
        rows = self._get_selected_rows()
        sumobj = self.PARAMCLASS()
        sumobj.title = "*".join(["%s%03d" % (self.PREFIX, row) for row in rows])
        try:
            for row in rows:
                obj = self.objects[row]
                if sumobj.data is None:
                    sumobj.copy_data_from(obj)
                else:
                    sumobj.data *= obj.data
        except Exception, msg:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent(), APP_NAME,
                                 _(u"Error:")+"\n%s" % str(msg))
            return
        self.add_object(sumobj)
    
    def compute_difference(self):
        rows = self._get_selected_rows()
        diffobj = self.PARAMCLASS()
        diffobj.title = "-".join(["%s%03d" % (self.PREFIX, row)
                                  for row in rows])
        try:
            obj0, obj1 = self.objects[rows[0]], self.objects[rows[1]]
            diffobj.copy_data_from(obj0)
            diffobj.data = obj1.data-obj0.data
        except Exception, msg:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent(), APP_NAME,
                                 _(u"Error:")+"\n%s" % str(msg))
            return
        self.add_object(diffobj)
    
    def compute_division(self):
        rows = self._get_selected_rows()
        diffobj = self.PARAMCLASS()
        diffobj.title = "/".join(["%s%03d" % (self.PREFIX, row)
                                  for row in rows])
        try:
            obj0, obj1 = self.objects[rows[0]], self.objects[rows[1]]
            diffobj.copy_data_from(obj0)
            diffobj.data = obj1.data/obj0.data
        except Exception, msg:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent(), APP_NAME,
                                 _(u"Error:")+"\n%s" % str(msg))
            return
        self.add_object(diffobj)
            
    #------Data Processing
    def apply_11_func(self, obj, orig, func):
        obj.data = func(orig.data)
    
    def compute_11(self, name, func):
        rows = self._get_selected_rows()
        for row in rows:
            orig = self.objects[row]
            obj = self.PARAMCLASS()
            obj.title = "%s(s%03d)" % (name, row)
            obj.copy_data_from(orig)
            try:
                self.apply_11_func(obj, orig, func)
            except Exception, msg:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self.parent(), APP_NAME,
                                     _(u"Error:")+"\n%s" % str(msg))
                return
            self.add_object(obj)
        
class SignalFT(ObjectFT):
    PARAMCLASS = SignalParam
    PREFIX = "s"
    #------ObjectFT API
    def setup(self, toolbar):
        super(SignalFT, self).setup(toolbar)
        
        # File actions
        new_action = create_action(self, _("New signal..."),
                                   icon=get_icon('filenew.png'),
                                   tip=_("Create a new signal"),
                                   triggered=self.new_signal)
        open_action = create_action(self, _("Open signal..."),
                                    icon=get_icon('fileopen.png'),
                                    tip=_("Open a signal"),
                                    triggered=self.open_signal)
        save_action = create_action(self, _("Save signal..."),
                                    icon=get_icon('filesave.png'),
                                    tip=_("Save selected signal"),
                                    triggered=self.save_signal)
        self.actlist_1more += [save_action]
        self.file_actions = [new_action, open_action, save_action]
        
        # Processing actions
        gaussian_action = create_action(self, _("Gaussian filter"),
                                        triggered=self.compute_gaussian)
        wiener_action = create_action(self, _("Wiener filter"),
                                      triggered=self.compute_wiener)
        fft_action = create_action(self, _("FFT"),
                                   tip=_("Warning: only real part is plotted"),
                                   triggered=self.compute_fft)
        ifft_action = create_action(self, _("Inverse FFT"),
                                   tip=_("Warning: only real part is plotted"),
                                    triggered=self.compute_ifft)
        self.actlist_1more += [gaussian_action, wiener_action,
                               fft_action, ifft_action]
        self.processing_actions = [gaussian_action, wiener_action, fft_action,
                                   ifft_action]
                                   
        add_actions(toolbar, [new_action, open_action, save_action])

    def make_item(self, row):
        signal = self.objects[row]
        x, y = signal.xydata
        item = make.mcurve(x, y.real, label=signal.title)
        self.items[row] = item
        return item
        
    def update_item(self, row):
        signal = self.objects[row]
        x, y = signal.xydata
        item = self.items[row]
        item.set_data(x, y.real)
        item.curveparam.label = signal.title
        
    #------Signal Processing
    def apply_11_func(self, obj, orig, func):
        xor, yor = orig.xydata
        obj.xydata = func(xor, yor)
    
    def compute_wiener(self):
        import scipy.signal as sps
        def func(x, y):
            return x, sps.wiener(y)
        self.compute_11("WienerFilter", func)
    
    def compute_gaussian(self):
        import scipy.ndimage as spi
        def func(x, y):
            return x, spi.gaussian_filter1d(y, 1.)
        self.compute_11("GaussianFilter", func)
                         
    def compute_fft(self):
        def func(x, y):
            y1 = np.fft.fft(y)
            x1 = np.fft.fftshift(np.fft.fftfreq(x.shape[-1], d=x[1]-x[0]))
            return x1, y1
        self.compute_11("FFT", func)
                         
    def compute_ifft(self):
        def func(x, y):
            y1 = np.fft.ifft(y)
            x1 = np.fft.fftshift(np.fft.fftfreq(x.shape[-1], d=x[1]-x[0]))
            return x1, y1
        self.compute_11("iFFT", func)
                            
    #------I/O
    def new_signal(self):
        """Create a new signal"""
        signalnew = SignalParamNew(title=_("Create a new signal"))
        rows = self._get_selected_rows()
        if rows:
            signalnew.size = len(self.objects[rows[-1]].data)
        signalnew.title = "%s %d" % (signalnew.title, self.number+1)
        if not signalnew.edit(parent=self.parent()):
            return
        self.number += 1
        signal = SignalParam()
        signal.title = signalnew.title
        xarr = np.linspace(signalnew.xmin, signalnew.xmax, signalnew.size)
        if signalnew.type == 'zeros':
            signal.xydata = np.vstack((xarr, np.zeros(signalnew.size)))
        elif signalnew.type == 'rand':
            signal.xydata = np.vstack((xarr, np.random.rand(signalnew.size)-.5))
        elif signalnew.type == 'gauss':
            class GaussParam(DataSet):
                a = FloatItem("Norm", default=1.)
                x0 = FloatItem("X0", default=0.0)
                sigma = FloatItem(u"σ", default=5.)
            param = GaussParam(_("New gaussian function"))
            if not param.edit(parent=self.parent()):
                return
            ygauss = param.a*np.exp(-.5*((xarr-param.x0)/param.sigma)**2)
            signal.xydata = np.vstack((xarr, ygauss))
        self.add_object(signal)
    
    def open_signal(self):
        """Open signal file"""
        saved_in, saved_out, saved_err = sys.stdin, sys.stdout, sys.stderr
        sys.stdout = None
        filename = QFileDialog.getOpenFileName(self.parent(), _("Open"),
                self.directory, 'Text files (*.txt *.csv)\nNumPy files (*.npy)')
        sys.stdin, sys.stdout, sys.stderr = saved_in, saved_out, saved_err
        if filename:
            filename = unicode(filename)
            self.directory = osp.dirname(filename)
            signal = SignalParam()
            signal.title = filename
            try:
                if osp.splitext(filename)[1] == ".npy":
                    xydata =np.load(filename)
                else:
                    for delimiter in ('\t', ',', ' ', ';'):
                        try:
                            xydata = np.loadtxt(filename, delimiter=delimiter)
                            break
                        except ValueError:
                            continue
                    else:
                        raise
            except Exception, msg:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self.parent(), APP_NAME,
                     (_(u"%s could not be opened:") % osp.basename(filename))+\
                     "\n"+str(msg))
                return
            if len(xydata.shape) == 1:
                xydata = np.vstack( (np.arange(xydata.size), xydata) )
            elif len(xydata.shape) == 2:
                rows, cols = xydata.shape
                if cols == 2 and rows > 2:
                    xydata = xydata.T
            signal.xydata = xydata
            self.add_object(signal)
            
    def save_signal(self):
        """Save selected signal"""
        rows = self._get_selected_rows()
        for row in rows:
            filename = QFileDialog.getSaveFileName(self, _("Save as"), 
                                   self.directory, _(u"CSV files")+" (*.csv)")
            if not filename:
                return
            filename = unicode(filename)
            self.directory = osp.dirname(filename)
            obj = self.objects[row]
            try:
                np.savetxt(filename, obj.xydata, delimiter=',')
            except Exception, msg:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self.parent(), APP_NAME,
                     (_(u"%s could not be written:") % osp.basename(filename))+\
                     "\n"+str(msg))
                return
        
class ImageFT(ObjectFT):
    PARAMCLASS = ImageParam
    PREFIX = "i"
    #------ObjectFT API
    def setup(self, toolbar):
        super(ImageFT, self).setup(toolbar)
        
        # File actions
        new_action = create_action(self, _("New image..."),
                                   icon=get_icon('filenew.png'),
                                   tip=_("Create a new image"),
                                   triggered=self.new_image)
        open_action = create_action(self, _("Open image..."),
                                    icon=get_icon('fileopen.png'),
                                    tip=_("Open an image"),
                                    triggered=self.open_image)
        save_action = create_action(self, _("Save image..."),
                                    icon=get_icon('filesave.png'),
                                    tip=_("Save selected image"),
                                    triggered=self.save_image)
        self.actlist_1more += [save_action]
        self.file_actions = [new_action, open_action, save_action]
        
        # Processing actions
        gaussian_action = create_action(self, _("Gaussian filter"),
                                        triggered=self.compute_gaussian)
        wiener_action = create_action(self, _("Wiener filter"),
                                      triggered=self.compute_wiener)
        fft_action = create_action(self, _("FFT"),
                                   tip=_("Warning: only real part is plotted"),
                                   triggered=self.compute_fft)
        ifft_action = create_action(self, _("Inverse FFT"),
                                   tip=_("Warning: only real part is plotted"),
                                    triggered=self.compute_ifft)
        self.actlist_1more += [gaussian_action, wiener_action,
                               fft_action, ifft_action]
        self.processing_actions = [gaussian_action, wiener_action, fft_action,
                                   ifft_action]
                                   
        add_actions(toolbar, [new_action, open_action, save_action])
        
    def make_item(self, row):
        image = self.objects[row]
        item = make.image(image.data.real, title=image.title, colormap='gray')
        self.items[row] = item
        return item
        
    def update_item(self, row):
        image = self.objects[row]
        item = self.items[row]
        item.set_data(image.data.real)
        item.imageparam.label = image.title
        
    #------Signal Processing
    def compute_wiener(self):
        import scipy.signal as sps
        self.compute_11("WienerFilter", sps.wiener)
    
    def compute_gaussian(self):
        import scipy.ndimage as spi
        self.compute_11("GaussianFilter",
                         lambda x: spi.gaussian_filter(x, 1.))
                         
    def compute_fft(self):
        self.compute_11("FFT", np.fft.fft2)
                         
    def compute_ifft(self):
        self.compute_11("iFFT", np.fft.ifft2)
                            
    #------I/O
    def new_image(self):
        """Create a new image"""
        imagenew = ImageParamNew(title=_("Create a new image"))
        rows = self._get_selected_rows()
        if rows:
            data = self.objects[rows[-1]].data
            imagenew.width = data.shape[1]
            imagenew.height = data.shape[0]
        imagenew.title = "%s %d" % (imagenew.title, self.number+1)
        if not imagenew.edit(parent=self.parent()):
            return
        self.number += 1
        image = ImageParam()
        image.title = imagenew.title
        shape = (imagenew.height, imagenew.width)
        dtype = imagenew.dtype
        if imagenew.type == 'zeros':
            image.data = np.zeros(shape, dtype=dtype)
        elif imagenew.type == 'empty':
            image.data = np.empty(shape, dtype=dtype)
        elif imagenew.type == 'rand':
            data = np.random.rand(*shape)
            from guiqwt.io import set_dynamic_range_from_dtype
            image.data = set_dynamic_range_from_dtype(data, dtype)
        self.add_object(image)
    
    def open_image(self):
        """Open image file"""
        saved_in, saved_out, saved_err = sys.stdin, sys.stdout, sys.stderr
        sys.stdout = None
        filename = QFileDialog.getOpenFileName(self.parent(), _("Open"),
            self.directory, 'Images (*.png *.jpg *.gif *.tif *.tiff *.dcm)')
        sys.stdin, sys.stdout, sys.stderr = saved_in, saved_out, saved_err
        if filename:
            filename = unicode(filename)
            self.directory = osp.dirname(filename)
            image = ImageParam()
            image.title = filename
            from guiqwt.io import imagefile_to_array
            try:
                image.data = imagefile_to_array(filename, to_grayscale=True)
            except Exception, msg:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self.parent(), APP_NAME,
                     (_(u"%s could not be opened:") % osp.basename(filename))+\
                     "\n"+str(msg))
                return
            self.add_object(image)
            
    def save_image(self):
        """Save selected image"""
        rows = self._get_selected_rows()
        for row in rows:
            filename = QFileDialog.getSaveFileName(self, _("Save as"), 
                     self.directory, 'Images (*.png *.jpg *.gif *.tif *.tiff)')
            if not filename:
                return
            filename = unicode(filename)
            self.directory = osp.dirname(filename)
            obj = self.objects[row]
            try:
                from guiqwt.io import array_to_imagefile
                array_to_imagefile(obj.data, filename)
            except Exception, msg:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self.parent(), APP_NAME,
                     (_(u"%s could not be written:") % osp.basename(filename))+\
                     "\n"+str(msg))
                return
        

class DockablePlotWidget(DockableWidget):
    LOCATION = Qt.RightDockWidgetArea
    def __init__(self, parent, plotwidgetclass, title, toolbar):
        super(DockablePlotWidget, self).__init__(parent)
        self.title = title
        self.toolbar = toolbar
        layout = QVBoxLayout()
        self.plotwidget = plotwidgetclass()
        layout.addWidget(self.plotwidget)
        self.setLayout(layout)
        self.setup()
        
    def get_plot(self):
        return self.plotwidget.plot
        
    def setup(self):
        title = unicode(self.toolbar.windowTitle())
        self.plotwidget.add_toolbar(self.toolbar, title)
        if isinstance(self.plotwidget, ImageWidget):
            self.plotwidget.register_all_image_tools()
        else:
            self.plotwidget.register_all_curve_tools()
        
    #------DockableWidget API
    def get_widget_title(self):
        """Return widget title"""
        return self.title
        
    def visibility_changed(self, enable):
        """DockWidget visibility has changed"""
        self.toolbar.setVisible(enable)
        

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setWindowIcon(get_icon('sift.svg'))
        self.setWindowTitle(APP_NAME)
                
        # Welcome message in statusbar:
        status = self.statusBar()
        status.showMessage(_("Welcome to %s!") % APP_NAME, 5000)

        self.signal_toolbar = self.addToolBar(_("Signal Processing Toolbar"))
        self.image_toolbar = self.addToolBar(_("Image Processing Toolbar"))

        # Signals
        curveplot_toolbar = self.addToolBar(_("Curve Plotting Toolbar"))
        self.curvewidget = DockablePlotWidget(self, CurveWidget,
                                              _("Curve plotting panel"),
                                              curveplot_toolbar)
        curveplot = self.curvewidget.get_plot()
        curveplot.add_item(make.legend("TR"))
        self.signalft = SignalFT(self, plot=curveplot)
        self.signalft.setup(self.signal_toolbar)
        
        # Images
        imagevis_toolbar = self.addToolBar(_("Image Visualization Toolbar"))
        self.imagewidget = DockablePlotWidget(self, ImageWidget,
                                              _("Image visualization panel"),
                                              imagevis_toolbar)
        self.imageft = ImageFT(self, self.imagewidget.get_plot())
        self.imageft.setup(self.image_toolbar)
        
        # Main window widgets
        self.tabwidget = QTabWidget()
        self.tabwidget.addTab(self.signalft, get_icon('curve.png'),
                              _("Signals"))
        self.tabwidget.addTab(self.imageft, get_icon('image.png'),
                              _("Images"))
        self.setCentralWidget(self.tabwidget)
        self.curve_dock = self.add_dockwidget(self.curvewidget)
        self.image_dock = self.add_dockwidget(self.imagewidget)
        self.tabifyDockWidget(self.curve_dock, self.image_dock)
        self.connect(self.tabwidget, SIGNAL('currentChanged(int)'),
                     self.tab_index_changed)
        self.connect(self.signalft, SIGNAL('object_added()'),
                     lambda: self.tabwidget.setCurrentIndex(0))
        self.connect(self.imageft, SIGNAL('object_added()'),
                     lambda: self.tabwidget.setCurrentIndex(1))
        
        # File menu
        self.quit_action = create_action(self, _("Quit"), shortcut="Ctrl+Q",
                                    icon=get_std_icon("DialogCloseButton"),
                                    tip=_("Quit application"),
                                    triggered=self.close)
        self.file_menu = self.menuBar().addMenu(_("File"))
        self.connect(self.file_menu, SIGNAL("aboutToShow()"),
                     self.update_file_menu)
        
        # Edit menu
        self.edit_menu = self.menuBar().addMenu(_("&Edit"))
        self.connect(self.edit_menu, SIGNAL("aboutToShow()"),
                     self.update_edit_menu)
        
        # Operation menu
        self.operation_menu = self.menuBar().addMenu(_("Operations"))
        self.connect(self.operation_menu, SIGNAL("aboutToShow()"),
                     self.update_operation_menu)
        
        # Processing menu
        self.proc_menu = self.menuBar().addMenu(_("Processing"))
        self.connect(self.proc_menu, SIGNAL("aboutToShow()"),
                     self.update_proc_menu)
        
        # View menu
        self.view_menu = view_menu = self.createPopupMenu()
        view_menu.setTitle(_(u"&View"))
        self.menuBar().addMenu(view_menu)
        
        # Help menu
        help_menu = self.menuBar().addMenu("?")
        about_action = create_action(self, _("About..."),
                                     icon=get_std_icon('MessageBoxInformation'),
                                     triggered=self.about)
        add_actions(help_menu, (about_action,))
        
        # Update selection dependent actions
        self.update_actions()
        
        # Show main window and raise the signal plot panel
        self.show()
        self.curve_dock.raise_()
                
    #------GUI refresh/setup
    def add_dockwidget(self, child):
        """Add QDockWidget and toggleViewAction"""
        dockwidget, location = child.create_dockwidget()
        self.addDockWidget(location, dockwidget)
        return dockwidget
        
    def update_actions(self):
        self.signalft.selection_changed()
        self.imageft.selection_changed()
        is_signal = self.tabwidget.currentWidget() is self.signalft
        self.signal_toolbar.setVisible(is_signal)
        self.image_toolbar.setVisible(not is_signal)
        
    def tab_index_changed(self, index):
        dock = (self.curve_dock, self.image_dock)[index]
        dock.raise_()
        self.update_actions()

    def update_file_menu(self):        
        self.file_menu.clear()
        objectft = self.tabwidget.currentWidget()
        actions = objectft.file_actions+[None, self.quit_action]
        add_actions(self.file_menu, actions)

    def update_edit_menu(self):        
        self.edit_menu.clear()
        objectft = self.tabwidget.currentWidget()
        add_actions(self.edit_menu, objectft.edit_actions)
        
    def update_operation_menu(self):
        self.operation_menu.clear()
        objectft = self.tabwidget.currentWidget()
        add_actions(self.operation_menu, objectft.operation_actions)
        
    def update_proc_menu(self):
        self.proc_menu.clear()
        objectft = self.tabwidget.currentWidget()
        add_actions(self.proc_menu, objectft.processing_actions)
                    
    #------?
    def about(self):
        QMessageBox.about( self, _("About ")+APP_NAME,
              """<b>%s</b> v%s<br>%s<p>%s Pierre Raybaut
              <br>Copyright &copy; 2010 CEA
              <p>Python %s, Qt %s, PyQt %s %s %s""" % \
              (APP_NAME, VERSION, APP_DESC, _("Developped by"),
               platform.python_version(),
               QT_VERSION_STR, PYQT_VERSION_STR, _("on"), platform.system()) )
        
        
if __name__ == '__main__':
    from guidata import qapplication
    app = qapplication()
    window = MainWindow()
    window.show()
    app.exec_()
