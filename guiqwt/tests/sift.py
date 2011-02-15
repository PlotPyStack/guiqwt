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
                         QFileDialog)
from PyQt4.QtCore import (QSize, QT_VERSION_STR, PYQT_VERSION_STR, Qt, SIGNAL,
                          QObject)

import sys, platform, os.path as osp
import numpy as np

from guidata.dataset.datatypes import DataSet, GetAttrProp
from guidata.dataset.dataitems import (IntItem, FloatArrayItem, StringItem,
                                       ChoiceItem, FloatItem)
from guidata.dataset.qtwidgets import DataSetEditGroupBox
from guidata.configtools import get_icon
from guidata.qthelpers import create_action, add_actions, get_std_icon
from guidata.utils import update_dataset

from guiqwt.config import _
from guiqwt.plot import CurveWidget, ImageWidget
from guiqwt.builder import make

APP_NAME = _("Sift")
APP_DESC = _("""Signal and Image Filtering Tool<br>
Simple signal and image processing application based on guiqwt and guidata""")
VERSION = '0.2.0'


class SignalParam(DataSet):
    _hide_data = False
    _hide_size = True
    title = StringItem(_("Title"), default=_("Untitled"))
    data = FloatArrayItem(_("Data"), transpose=True, minmax="rows",
                          ).set_prop("display", hide=GetAttrProp("_hide_data"))
    xmin = FloatItem("Xmin", default=-10.).set_prop("display",
                                                hide=GetAttrProp("_hide_size"))
    xmax = FloatItem("Xmax", default=10.).set_prop("display",
                                                hide=GetAttrProp("_hide_size"))
    size = IntItem(_("Size"), help=_("Signal size (total number of points)"),
                   min=1, default=500).set_prop("display",
                                                hide=GetAttrProp("_hide_size"))

class SignalParamNew(SignalParam):
    _hide_data = True
    _hide_size = False
    type = ChoiceItem(_("Type"),
                      (("rand", _("random")), ("zeros", _("zeros")),
                       ("gauss", _("gaussian"))))


class ObjectHandler(QObject):
    """Object handling the item list, the selected item properties and plot"""
    PARAMCLASS = None
    PREFIX = None
    def __init__(self, parent):
        super(ObjectHandler, self).__init__(parent)
        self.objects = [] # signals or images
        self.items = [] # associated plot items
        self.listwidget = None
        self.properties = None
        self.plotwidget = None
        self._hsplitter = None
        
        self.number = 0

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
        
        self.setup_plotwidget(toolbar)
        
    def setup_plotwidget(self, toolbar):
        raise NotImplementedError
    
    def create_hsplitter(self):
        self._hsplitter = QSplitter()
        self._hsplitter.addWidget(self.listwidget)
        self._hsplitter.addWidget(self.properties)
        return self._hsplitter
        
    def get_operation_actions(self):
        raise NotImplementedError
        
    def get_processing_actions(self):
        raise NotImplementedError

    #------GUI refresh/setup
    def current_item_changed(self, row):
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
        plot = self.plotwidget.plot
        for item in self.items:
            if item is not None:
                item.hide()
        for row in self._get_selected_rows():
            item = self.items[row]
            if item is None:
                item = self.make_item(row)
                plot.add_item(item)
            else:
                self.update_item(row)
                plot.set_item_visible(item, True)
        plot.do_autoscale()
        
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
        
class SignalHandler(ObjectHandler):
    PARAMCLASS = SignalParam
    PREFIX = "s"
    def get_operation_actions(self):
        sum_action = create_action(self, _("Sum"), triggered=self.compute_sum)
        diff_action = create_action(self, _("Difference"),
                                    triggered=self.compute_difference)
        self.actlist_2more += [sum_action]
        self.actlist_2 += [diff_action]
        return sum_action, diff_action
        
    def get_processing_actions(self):
        gaussian_action = create_action(self, _("Gaussian filter"),
                                        triggered=self.compute_gaussian)
        wiener_action = create_action(self, _("Wiener filter"),
                                      triggered=self.compute_wiener)
        self.actlist_1more += [gaussian_action, wiener_action]
        return gaussian_action, wiener_action
        
    def setup_plotwidget(self, toolbar):
        self.plotwidget = CurveWidget()
        self.plotwidget.add_toolbar(toolbar, "default")
        self.plotwidget.register_all_curve_tools()
        self.plotwidget.plot.add_item(make.legend("TR"))

    def make_item(self, row):
        signal = self.objects[row]
        x, y = signal.data
        item = make.mcurve(x, y, label=signal.title)
        self.items[row] = item
        return item
        
    def update_item(self, row):
        signal = self.objects[row]
        x, y = signal.data
        item = self.items[row]
        item.set_data(x, y)
        item.curveparam.label = signal.title
        
    #------Operations
    def compute_sum(self):
        rows = self._get_selected_rows()
        signal = SignalParam()
        signal.title = "+".join(["s%03d" % row for row in rows])
        signal.size = self.objects[rows[0]].size
        try:
            for row in rows:
                sig = self.objects[row]
                if signal.data is None:
                    signal.data = np.array(sig.data, copy=True)
                else:
                    signal.data[1] += sig.data[1]
        except Exception, msg:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent(), APP_NAME,
                                 _(u"Error:")+"\n%s" % str(msg))
            return
        self.add_object(signal)
    
    def compute_difference(self):
        rows = self._get_selected_rows()
        signal = SignalParam()
        signal.title = "-".join(["s%03d" % row for row in rows])
        signal.size = self.objects[rows[0]].size
        try:
            sig0, sig1 = self.objects[rows[0]], self.objects[rows[1]]
            signal.data = np.array(sig0.data, copy=True)
            signal.data[1] = sig1.data[1]-sig0.data[1]
        except Exception, msg:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent(), APP_NAME,
                                 _(u"Error:")+"\n%s" % str(msg))
            return
        self.add_object(signal)
            
    #------Signal Processing
    def _compute_11(self, name, func):
        rows = self._get_selected_rows()
        for row in rows:
            orig = self.objects[row]
            signal = SignalParam()
            signal.title = "%s(s%03d)" % (name, row)
            signal.size = orig.size
            signal.data = orig.data.copy()
            try:
                signal.data[1] = func(orig.data[1])
            except Exception, msg:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self.parent(), APP_NAME,
                                     _(u"Error:")+"\n%s" % str(msg))
                return
            self.add_object(signal)
    
    def compute_wiener(self):
        import scipy.signal as sps
        self._compute_11("WienerFilter", sps.wiener)
    
    def compute_gaussian(self):
        import scipy.ndimage as spi
        self._compute_11("GaussianFilter",
                         lambda x: spi.gaussian_filter1d(x, 1.))
                            
    #------I/O
    def new_signal(self):
        """Create a new signal"""
        signalnew = SignalParamNew(title=_("Create a new signal"))
        signalnew.title = "%s %d" % (signalnew.title, self.number+1)
        if not signalnew.edit(parent=self.parent()):
            return
        self.number += 1
        signal = SignalParam()
        signal.title = signalnew.title
        xarr = np.linspace(signalnew.xmin, signalnew.xmax, signalnew.size)
        if signalnew.type == 'zeros':
            signal.data = np.vstack((xarr, np.zeros(signalnew.size)))
        elif signalnew.type == 'rand':
            signal.data = np.vstack((xarr, np.random.rand(signalnew.size)-.5))
        elif signalnew.type == 'gauss':
            class GaussParam(DataSet):
                a = FloatItem("Norm", default=1.)
                x0 = FloatItem("X0", default=0.0)
                sigma = FloatItem(u"σ", default=5.)
            param = GaussParam(_("New gaussian function"))
            if not param.edit(parent=self.parent()):
                return
            ygauss = param.a*np.exp(-.5*((xarr-param.x0)/param.sigma)**2)
            signal.data = np.vstack((xarr, ygauss))
        self.add_object(signal)
    
    def open_signal(self):
        """Open signal file"""
        saved_in, saved_out, saved_err = sys.stdin, sys.stdout, sys.stderr
        sys.stdout = None
        filename = QFileDialog.getOpenFileName(self.parent(), _("Open"), "",
                           'Text files (*.txt *.csv)\nNumPy files (*.npy)')
        sys.stdin, sys.stdout, sys.stderr = saved_in, saved_out, saved_err
        if filename:
            filename = unicode(filename)
            signal = SignalParam()
            signal.title = filename
            try:
                if osp.splitext(filename)[1] == ".npy":
                    data =np.load(filename)
                else:
                    for delimiter in ('\t', ',', ' ', ';'):
                        try:
                            data = np.loadtxt(filename, delimiter=delimiter)
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
            if len(data.shape) == 1:
                data = np.vstack( (np.arange(data.size), data) )
            elif len(data.shape) == 2:
                rows, cols = data.shape
                if cols == 2 and rows > 2:
                    data = data.T
            signal.data = data
            signal.size = data.shape[1] # or data.size/2
            self.add_object(signal)
        
class ImageHandler(ObjectHandler):
    PARAMCLASS = None
    PREFIX = "i"
    def setup_plotwidget(self, toolbar):
        self.plotwidget = ImageWidget(self)
        self.plotwidget.add_toolbar(toolbar, "default")
        self.plotwidget.register_all_image_tools()
        

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setWindowIcon(get_icon('sift.svg'))
        self.setWindowTitle(APP_NAME)
        self.resize(QSize(600, 800))
        
        main_toolbar = self.addToolBar("Main")
        
        # Welcome message in statusbar:
        status = self.statusBar()
        status.showMessage(_("Welcome to %s!") % APP_NAME, 5000)

        # Signals
        self.signalhandler = SignalHandler(self)
        toolbar = self.addToolBar("Signal")
        self.signalhandler.setup(toolbar)
        hsplitter = self.signalhandler.create_hsplitter()
        self.plotwidget = self.signalhandler.plotwidget
        
        # File menu
        file_menu = self.menuBar().addMenu(_("File"))
        new_action = create_action(self, _("New..."), shortcut="Ctrl+N",
                                   icon=get_icon('filenew.png'),
                                   tip=_("Create a new signal"),
                                   triggered=self.signalhandler.new_signal)
        open_action = create_action(self, _("Open..."), shortcut="Ctrl+O",
                                    icon=get_icon('fileopen.png'),
                                    tip=_("Open a signal"),
                                    triggered=self.signalhandler.open_signal)
        quit_action = create_action(self, _("Quit"), shortcut="Ctrl+Q",
                                    icon=get_std_icon("DialogCloseButton"),
                                    tip=_("Quit application"),
                                    triggered=self.close)
        add_actions(file_menu, (new_action, open_action, None, quit_action))
        add_actions(main_toolbar, (new_action, open_action, ))

        # Operation menu
        operation_menu = self.menuBar().addMenu(_("Operations"))
        add_actions(operation_menu, self.signalhandler.get_operation_actions())
        
        # Processing menu
        proc_menu = self.menuBar().addMenu(_("Processing"))
        add_actions(proc_menu, self.signalhandler.get_processing_actions())
        
        # Help menu
        help_menu = self.menuBar().addMenu("?")
        about_action = create_action(self, _("About..."),
                                     icon=get_std_icon('MessageBoxInformation'),
                                     triggered=self.about)
        add_actions(help_menu, (about_action,))
        
        vsplitter = QSplitter(Qt.Vertical, self)
        vsplitter.setContentsMargins(10, 10, 10, 10)
        self.setCentralWidget(vsplitter)
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(self.plotwidget)
        vsplitter.setStretchFactor(0, 0)
        vsplitter.setStretchFactor(1, 1)
        vsplitter.setHandleWidth(10)
        vsplitter.setSizes([1, 2])
        
        self.signalhandler.selection_changed() # Update selection dependent actions
                
    #------GUI refresh/setup
    def add_dockwidget(self, child):
        """Add QDockWidget and toggleViewAction"""
        dockwidget, location = child.create_dockwidget()
        self.addDockWidget(location, dockwidget)
        return dockwidget
                    
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
