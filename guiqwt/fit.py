# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.fit
----------

The `fit` module provides an interactive curve fitting tool allowing:
    * to fit data manually (by moving sliders)
    * or automatically (with standard optimization algorithms 
      provided by :py:mod:`scipy`).

Example
~~~~~~~

.. literalinclude:: ../guiqwt/tests/fit.py
   :start-after: SHOW
   :end-before: Workaround for Sphinx v0.6 bug: empty 'end-before' directive

.. image:: images/screenshots/fit.png

Reference
~~~~~~~~~

.. autofunction:: guifit

.. autoclass:: FitDialog
   :members:
   :inherited-members:
.. autoclass:: FitParam
   :members:
   :inherited-members:
.. autoclass:: AutoFitParam
   :members:
   :inherited-members:
"""

from PyQt4.QtGui import (QGridLayout, QLabel, QSlider, QPushButton,
                         QCheckBox, QDialog, QVBoxLayout, QHBoxLayout, QWidget,
                         QDialogButtonBox)
from PyQt4.QtCore import Qt, SIGNAL, QObject, SLOT

import numpy as np
from numpy import inf # Do not remove this import (used by optimization funcs)

import guidata
from guidata.qthelpers import create_groupbox
from guidata.configtools import get_icon
from guidata.dataset.datatypes import DataSet
from guidata.dataset.dataitems import (StringItem, FloatItem, IntItem,
                                       ChoiceItem, BoolItem)

# Local imports
from guiqwt.config import _
from guiqwt.builder import make
from guiqwt.plot import CurveWidgetMixin
from guiqwt.signals import SIG_RANGE_CHANGED

class AutoFitParam(DataSet):
    xmin = FloatItem("xmin")
    xmax = FloatItem("xmax")
    method = ChoiceItem(_("Method"),
                        [ ("simplex", "Simplex"), ("powel", "Powel"),
                          ("bfgs", "BFGS"), ("l_bfgs_b", "L-BFGS-B"),
                          ("cg", _("Conjugate Gradient")),
                          ("lq", _("Least squares")), ],
                        default="lq")
    err_norm = StringItem("enorm", default=2.0,
                          help=_("for simplex, powel, cg and bfgs norm used "
                                 "by the error function"))
    xtol = FloatItem("xtol", default=0.0001,
                     help=_("for simplex, powel, least squares"))
    ftol = FloatItem("ftol", default=0.0001,
                     help=_("for simplex, powel, least squares"))
    gtol = FloatItem("gtol", default=0.0001, help=_("for cg, bfgs"))
    norm = StringItem("norm", default="inf",
                      help=_("for cg, bfgs. inf is max, -inf is min"))


class FitParam(DataSet):
    name = StringItem(_("Name"))
    value = FloatItem(_("Value"), default=0.0)
    min = FloatItem(_("Min"), default=-1.0)
    max = FloatItem(_("Max"), default=1.0).set_pos(col=1)
    steps = IntItem(_("Steps"), default=5000)
    format = StringItem(_("Format"), default="%.3f").set_pos(col=1)
    logscale = BoolItem(_("Logarithmic"), _("Scale"))

    def __init__(self, name, value, min, max, logscale=False,
                 steps=5000, format='%.3f'):
        DataSet.__init__(self, title=_("Curve fitting parameter"))
        self.name = name
        self.value = value
        self.min = min
        self.max = max
        self.logscale = logscale
        self.steps = steps
        self.format = format
        self.button = None
        self.checkbox = None
        self.slider = None
        self.label = None
        
    def create_widgets(self, parent):
        self.button = QPushButton(get_icon('edit.png'), '', parent)
        self.button.setToolTip(
                        _("Edit '%s' fit parameter properties") % self.name)
        QObject.connect(self.button, SIGNAL('clicked()'), self.edit_param)
        self.checkbox = QCheckBox('Log', parent)
        self.checkbox.setToolTip(_('Logarithmic scale'))
        self.update_checkbox_state()
        QObject.connect(self.checkbox, SIGNAL('stateChanged(int)'),
                        self.set_scale)
        self.label = QLabel(parent)
        self.slider = QSlider(parent)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setRange(0, self.steps-1)
        QObject.connect(self.slider, SIGNAL("valueChanged(int)"),
                        self.slider_value_changed)
        self.set_text()
        self.update()
        return self.button, self.checkbox, self.slider, self.label
        
    def get_widgets(self):
        return self.button, self.checkbox, self.slider, self.label
        
    def set_scale(self, state):
        self.logscale = state > 0
        self.update_slider_value()
        
    def set_text(self):
        if self.value is None:
            value_str = '-'
        else:
            value_str = self.format % self.value
        self.label.setText('<b>%s</b> : %s' % (self.name, value_str))
        
    def slider_value_changed(self, int_value):
        if self.logscale:
            min, max = np.log10(self.min), np.log10(self.max)
            self.value = 10**(min+(max-min)*int_value/self.steps)
        else:
            self.value = self.min+(self.max-self.min)*int_value/self.steps
        self.set_text()
    
    def _logscale_check(self):
        if self.min == 0:
            self.min = self.max/10
        if self.max == 0:
            self.max = self.mix*10
    
    def update_slider_value(self):
        if self.logscale:
            self._logscale_check()
            value, min, max = (np.log10(self.value), np.log10(self.min),
                               np.log10(self.max))
        else:
            value, min, max = self.value, self.min, self.max
        if value is None or min is None or max is None:
            self.slider.setEnabled(False)
        else:
            self.slider.setEnabled(True)
            intval = int(self.steps*(value-min)/(max-min))
            self.slider.blockSignals(True)
            self.slider.setValue(intval)
            self.slider.blockSignals(False)

    def update_checkbox_state(self):
        state = Qt.Checked if self.logscale else Qt.Unchecked
        self.checkbox.setCheckState(state)

    def edit_param(self):
        res = self.edit()
        if res:
            self.update_checkbox_state()
            self.update()

    def update(self):
        self.slider.setRange(0, self.steps-1)
        self.update_slider_value()
        self.set_text()


class FitWidgetMixin(CurveWidgetMixin):
    def __init__(self, wintitle="guiqwt plot", icon="guiqwt.png",
                 toolbar=False, options=None, panels=None, param_cols=1):
        if wintitle is None:
            wintitle = _('Curve fitting')
            
        self.x = None
        self.y = None
        self.fitfunc = None
        self.fitargs = None
        self.fitkwargs = None
        self.fitparams = None
        self.autofit_prm = None
        
        self.param_cols = param_cols
        self.button_layout = None
        self.button_list = [] # list of buttons to be disabled at startup

        self.params_layout = None
        
        CurveWidgetMixin.__init__(self, wintitle=wintitle, icon=icon, 
                                  toolbar=toolbar, options=options,
                                  panels=panels)
        
        self.xrange = None
        self.show_xrange = False
        
        self.refresh()
        
    # CurveWidgetMixin API -----------------------------------------------------
    def setup_widget_layout(self):
        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.toolbar)
        vlayout.addLayout(self.plot_layout)
        self.setLayout(vlayout)
        self.button_layout = self.create_button_layout()
        vlayout.addSpacing(10)
        vlayout.addLayout(self.button_layout)
        
    def create_plot(self, options):
        super(FitWidgetMixin, self).create_plot(options)
        for plot in self.get_plots():
            self.connect(plot, SIG_RANGE_CHANGED, self.range_changed)
        self.params_layout = QGridLayout()
        params_group = create_groupbox(self, _("Fit parameters"),
                                       layout=self.params_layout)
        self.plot_layout.addWidget(params_group, 1, 0)
        
    # Public API ---------------------------------------------------------------  
    def set_data(self, x, y, fitfunc=None, fitparams=None,
                 fitargs=None, fitkwargs=None):
        if self.fitparams is not None and fitparams is not None:
            self.clear_params_layout()
        self.x = x
        self.y = y
        if fitfunc is not None:
            self.fitfunc = fitfunc
        if fitparams is not None:
            self.fitparams = fitparams
        if fitargs is not None:
            self.fitargs = fitargs
        if fitkwargs is not None:
            self.fitkwargs = fitkwargs
        self.autofit_prm = AutoFitParam(title=_("Automatic fitting options"))
        self.autofit_prm.xmin = x.min()
        self.autofit_prm.xmax = x.max()
        self.compute_imin_imax()
        if self.fitparams is not None and fitparams is not None:
            self.populate_params_layout()
        self.refresh()
        
    def set_fit_data(self, fitfunc, fitparams, fitargs=None, fitkwargs=None):
        if self.fitparams is not None:
            self.clear_params_layout()
        self.fitfunc = fitfunc
        self.fitparams = fitparams
        self.fitargs = fitargs
        self.fitkwargs = fitkwargs
        self.populate_params_layout()
        self.refresh()
        
    def clear_params_layout(self):
        for i, param in enumerate(self.fitparams):
            for widget in param.get_widgets():
                self.params_layout.removeWidget(widget)
                widget.hide()
        
    def populate_params_layout(self):
        row_contents = []
        row_nb = 0
        col_nb = 0
        for i, param in enumerate(self.fitparams):
            button, checkbox, slider, label = param.create_widgets(self)
            self.connect(slider, SIGNAL("valueChanged(int)"), self.refresh)
            self.connect(checkbox, SIGNAL("stateChanged(int)"), self.refresh)
            row_contents += [(button,   row_nb, 0+col_nb*5),
                             (checkbox, row_nb, 1+col_nb*5),
                             (slider,   row_nb, 2+col_nb*5),
                             (label,    row_nb, 3+col_nb*5)]
            col_nb += 1
            if col_nb == self.param_cols:
                row_nb += 1
                col_nb = 0
                for widget, row, col in row_contents:
                    self.params_layout.addWidget(widget, row, col)
        if self.param_cols > 1:
            for col_nb in range(self.param_cols):
                self.params_layout.setColumnStretch(2+(col_nb-1)*5, 3)
                self.params_layout.setColumnStretch(4+(col_nb-1)*5, 1)
                self.params_layout.setColumnStretch(7+(col_nb-1)*5, 3)
    
    def create_button_layout(self):        
        btn_layout = QHBoxLayout()
        auto_button = QPushButton(get_icon('apply.png'), _("Auto"), self)
        self.connect(auto_button, SIGNAL("clicked()"), self.autofit)
        autoprm_button = QPushButton(get_icon('settings.png'),
                                     _("Fit parameters..."), self)
        self.connect(autoprm_button, SIGNAL("clicked()"), self.edit_parameters)
        xrange_button = QPushButton(get_icon('xrange.png'), _("Fit bounds"),
                                    self)
        xrange_button.setCheckable(True)
        self.connect(xrange_button, SIGNAL("toggled(bool)"), self.toggle_xrange)
        btn_layout.addWidget(auto_button)
        btn_layout.addStretch()
        btn_layout.addWidget(autoprm_button)
        btn_layout.addWidget(xrange_button)
        self.button_list += [auto_button, autoprm_button, xrange_button]
        return btn_layout
        
    def get_fitfunc_arguments(self):
        """Return fitargs and fitkwargs"""
        fitargs = self.fitargs
        if self.fitargs is None:
            fitargs = []
        fitkwargs = self.fitkwargs
        if self.fitkwargs is None:
            fitkwargs = {}
        return fitargs, fitkwargs
        
    def refresh(self):
        """Refresh Fit Tool dialog box"""
        # Update button states
        enable = self.x is not None and self.y is not None \
                 and self.x.size > 0 and self.y.size > 0 \
                 and self.fitfunc is not None and self.fitparams is not None \
                 and len(self.fitparams) > 0
        for btn in self.button_list:
            btn.setEnabled(enable)
            
        if not enable:
            # Fit widget is not yet configured
            return

        fitargs, fitkwargs = self.get_fitfunc_arguments()
        yfit = self.fitfunc(self.x, [p.value for p in self.fitparams],
                            *fitargs, **fitkwargs)
        self.xrange = make.range(self.autofit_prm.xmin, self.autofit_prm.xmax)
        self.xrange.setVisible(self.show_xrange)
        items = [make.curve(self.x, self.y, _("Data"), color="b", linewidth=2),
                 make.curve(self.x, yfit, _("Fit"), color="r", linewidth=2),
                 make.legend(), self.xrange]
        plot = self.get_plot()
        plot.del_all_items()
        for item in items:
            plot.add_item(item)
        plot.replot()
        plot.disable_autoscale()
        
    def range_changed(self, xrange_obj, xmin, xmax):
        self.autofit_prm.xmin, self.autofit_prm.xmax = xmin, xmax
        self.compute_imin_imax()
        
    def toggle_xrange(self, state):
        self.xrange.setVisible(state)
        plot = self.get_plot()
        plot.replot()
        if state:
            plot.set_active_item(self.xrange)
        self.show_xrange = state
        
    def edit_parameters(self):
        if self.autofit_prm.edit():
            self.xrange.set_range(self.autofit_prm.xmin, self.autofit_prm.xmax)
            plot = self.get_plot()
            plot.replot()
            self.compute_imin_imax()
        
    def compute_imin_imax(self):
        self.i_min = self.x.searchsorted(self.autofit_prm.xmin)
        self.i_max = self.x.searchsorted(self.autofit_prm.xmax)
        
    def errorfunc(self, params):
        x = self.x[self.i_min:self.i_max]
        y = self.y[self.i_min:self.i_max]
        fitargs, fitkwargs = self.get_fitfunc_arguments()
        return y - self.fitfunc(x, params, *fitargs, **fitkwargs)

    def autofit(self):
        meth = self.autofit_prm.method
        x0 = np.array([p.value for p in self.fitparams])
        if meth == "lq":
            x = self.autofit_lq(x0)
        elif meth=="simplex":
            x = self.autofit_simplex(x0)
        elif meth=="powel":
            x = self.autofit_powel(x0)
        elif meth=="bfgs":
            x = self.autofit_bfgs(x0)
        elif meth=="l_bfgs_b":
            x = self.autofit_l_bfgs(x0)
        elif meth=="cg":
            x = self.autofit_cg(x0)
        else:
            return
        for v,p in zip(x, self.fitparams):
            p.value = v
        self.refresh()
        for prm in self.fitparams:
            prm.update()

    def get_norm_func(self):
        prm = self.autofit_prm
        err_norm = eval(prm.err_norm)
        def func(params):
            err = np.linalg.norm(self.errorfunc(params), err_norm)
            return err
        return func

    def autofit_simplex(self, x0):
        prm = self.autofit_prm
        from scipy.optimize import fmin
        x = fmin(self.get_norm_func(), x0, xtol=prm.xtol, ftol=prm.ftol)
        return x

    def autofit_powel(self, x0):
        prm = self.autofit_prm
        from scipy.optimize import fmin_powell
        x = fmin_powell(self.get_norm_func(), x0, xtol=prm.xtol, ftol=prm.ftol)
        return x

    def autofit_bfgs(self, x0):
        prm = self.autofit_prm
        from scipy.optimize import fmin_bfgs
        x = fmin_bfgs(self.get_norm_func(), x0, gtol=prm.gtol,
                      norm=eval(prm.norm))
        return x

    def autofit_l_bfgs(self, x0):
        prm = self.autofit_prm
        bounds = [(p.min, p.max) for p in self.fitparams]
        from scipy.optimize import fmin_l_bfgs_b
        x, _f, _d = fmin_l_bfgs_b(self.get_norm_func(), x0, pgtol=prm.gtol,
                          approx_grad=1, bounds=bounds)
        return x
        
    def autofit_cg(self, x0):
        prm = self.autofit_prm
        from scipy.optimize import fmin_cg
        x = fmin_cg(self.get_norm_func(), x0, gtol=prm.gtol,
                    norm=eval(prm.norm))
        return x

    def autofit_lq(self, x0):
        prm = self.autofit_prm
        def func(params):
            err = self.errorfunc(params)
            return err
        from scipy.optimize import leastsq
        x, _ier = leastsq(func, x0, xtol=prm.xtol, ftol=prm.ftol)
        return x

    def get_values(self):
        """Convenience method to get fit parameter values"""
        return [param.value for param in self.fitparams]


class FitWidget(QWidget, FitWidgetMixin):
    def __init__(self, wintitle=None, icon="guiqwt.png", toolbar=False,
                 options=None, parent=None, panels=None,
                 param_cols=1):
        QWidget.__init__(self, parent)
        FitWidgetMixin.__init__(self, wintitle, icon, toolbar, options, panels,
                                param_cols)


class FitDialog(QDialog, FitWidgetMixin):
    def __init__(self, wintitle=None, icon="guiqwt.png", edit=True,
                 toolbar=False, options=None, parent=None, panels=None,
                 param_cols=1):
        QDialog.__init__(self, parent)
        self.edit = edit
        FitWidgetMixin.__init__(self, wintitle, icon, toolbar, options, panels,
                                param_cols)
        self.setWindowFlags(Qt.Window)
        
    def setup_widget_layout(self):
        FitWidgetMixin.setup_widget_layout(self)
        if self.edit:
            self.install_button_layout()
        
    def install_button_layout(self):
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.connect(bbox, SIGNAL("accepted()"), SLOT("accept()"))
        self.connect(bbox, SIGNAL("rejected()"), SLOT("reject()"))
        self.button_layout.addStretch()
        self.button_layout.addWidget(bbox)
        self.button_list += [bbox.button(QDialogButtonBox.Ok)]
        

def guifit(x, y, fitfunc, fitparams, fitargs=None, fitkwargs=None,
           wintitle=None, title=None, xlabel=None, ylabel=None, param_cols=1):
    """GUI-based curve fitting tool"""
    _app = guidata.qapplication()
#    widget = FitWidget(x, y, fitfunc, fitparams)
#    widget.show()
#    _app.exec_()
    dlg = FitDialog(edit=True, wintitle=wintitle, toolbar=True,
                    param_cols=param_cols,
                    options=dict(title=title, xlabel=xlabel, ylabel=ylabel))
    dlg.set_data(x, y, fitfunc, fitparams, fitargs, fitkwargs)
    
    if dlg.exec_():
        return dlg.get_values()


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y = np.cos(1.5*x)+np.random.rand(x.shape[0])*.2
    def fit(x, params):
        a, b = params
        return np.cos(b*x)+a
    a = FitParam("Offset", 1., 0., 2.)
    b = FitParam("Frequency", 2.001, 1., 10., logscale=True)
    params = [a, b]
    values = guifit(x, y, fit, params, param_cols=2)
    print values
    print [param.value for param in params]
