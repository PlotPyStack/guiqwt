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

from PyQt4.QtGui import (QGridLayout, QLabel, QSlider, QPushButton, QFrame,
                         QCheckBox)
from PyQt4.QtCore import Qt, SIGNAL, QObject

import numpy as np
from numpy import inf # Do not remove this import (used by optimization funcs)

import guidata
from guidata.configtools import get_icon
from guidata.dataset.datatypes import DataSet
from guidata.dataset.dataitems import (StringItem, FloatItem, IntItem,
                                       ChoiceItem, BoolItem)

# Local imports
from guiqwt.config import _
from guiqwt.builder import make
from guiqwt.plot import CurveDialog
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
    steps = IntItem(_("Steps"), default=500)
    format = StringItem(_("Format"), default="%.3f").set_pos(col=1)
    logscale = BoolItem(_("Logarithmic"), _("Scale"))

    def __init__(self, name, value, min, max, logscale=False,
                 steps=500, format='%.3f'):
        DataSet.__init__(self, title=_("Curve fitting parameter"))
        self.name = name
        self.value = value
        self.min = min
        self.max = max
        self.logscale = logscale
        self.steps = steps
        self.format = format
        self.label = None
        
    def get_widgets(self, parent):
        button = QPushButton(get_icon('edit.png'), _('Edit'), parent)
        button.setToolTip(_("Edit fit parameter '%s' properties") % self.name)
        QObject.connect(button, SIGNAL('clicked()'), self.edit_param)
        self.checkbox = QCheckBox(_('Logarithmic scale'), parent)
        self.update_checkbox_state()
        QObject.connect(self.checkbox, SIGNAL('stateChanged(int)'), self.set_scale)
        self.label = QLabel(parent)
        self.slider = QSlider(parent)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setRange(0, self.steps-1)
        QObject.connect(self.slider, SIGNAL("valueChanged(int)"),
                        self.slider_value_changed)
        self.set_text()
        self.update()
        return button, self.checkbox, self.slider, self.label
        
    def set_scale(self, state):
        self.logscale = state > 0
        self.update_slider_value()
        
    def set_text(self):
        text = ('<b>%s</b> : '+self.format) % (self.name, self.value)
        self.label.setText(text)
        
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
        intval = int(self.steps*(value-min)/(max-min))
        self.slider.setValue(intval)

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


class FitDialog(CurveDialog):
    def __init__(self, x, y, fitfunc, fitparams, title=None):
        if title is None:
            title = _('Curve fitting')
        self.x = x
        self.y = y
        self.fitfunc = fitfunc
        self.fitparams = fitparams
        super(FitDialog, self).__init__(wintitle=title, icon="guiqwt.png",
                                        edit=True, toolbar=True, options=None)
                
        self.autofit_prm = AutoFitParam(title=_("Automatic fitting options"))
        self.autofit_prm.xmin = x.min()
        self.autofit_prm.xmax = x.max()
        self.compute_imin_imax()
        
        self.xrange = None
        self.show_xrange = False
        self.refresh()
        
    # CurveDialog API ----------------------------------------------------------
    def install_button_layout(self):
        auto_button = QPushButton(get_icon('apply.png'), _("Auto"), self)
        self.connect(auto_button, SIGNAL("clicked()"), self.autofit)
        autoprm_button = QPushButton(get_icon('settings.png'),
                                     _("Fit parameters..."), self)
        self.connect(autoprm_button, SIGNAL("clicked()"), self.edit_parameters)
        xrange_button = QPushButton(get_icon('xrange.png'), _("Fit bounds"),
                                    self)
        xrange_button.setCheckable(True)
        self.connect(xrange_button, SIGNAL("toggled(bool)"), self.toggle_xrange)
        
        self.button_layout.addWidget(auto_button)
        self.button_layout.addWidget(autoprm_button)
        self.button_layout.addWidget(xrange_button)
        super(FitDialog, self).install_button_layout()

    def create_plot(self, options):
        super(FitDialog, self).create_plot(options)
        for plot in self.get_plots():
            self.connect(plot, SIG_RANGE_CHANGED, self.range_changed)
        
        params_frame = QFrame(self)
        params_frame.setFrameShape(QFrame.Box)
        params_frame.setFrameShadow(QFrame.Sunken)
        params_layout = QGridLayout()
        params_frame.setLayout(params_layout)
        for i, param in enumerate(self.fitparams):
            button, checkbox, slider, label = param.get_widgets(self)
            self.connect(slider, SIGNAL("valueChanged(int)"), self.refresh)
            self.connect(checkbox, SIGNAL("stateChanged(int)"), self.refresh)
            params_layout.addWidget(button, i, 0)

            params_layout.addWidget(checkbox, i, 1)
            params_layout.addWidget(slider, i, 2)
            params_layout.addWidget(label, i, 3)
        self.plot_layout.addWidget(params_frame, 1, 0)
        
    # Public API ---------------------------------------------------------------        
    def refresh(self):
        """Refresh Fit Tool dialog box"""
        yfit = self.fitfunc(self.x, [p.value for p in self.fitparams])
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
        return y - self.fitfunc(x, params)

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
        

def guifit(x, y, fitfunc, fitparams, title=None):
    """GUI-based curve fitting tool"""
    _app = guidata.qapplication()
    dlg = FitDialog(x, y, fitfunc, fitparams)
    if dlg.exec_():
        return dlg.get_values()


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    y = np.cos(1.5*x)+np.random.rand(x.shape[0])*.2
    def fit(x, params):
        a, b = params
        return np.cos(b*x)+a
    a = FitParam("Offset", 1., 0., 2.)
    b = FitParam("Frequency", 2., 1., 10., logscale=True)
    params = [a, b]
    values = guifit(x, y, fit, params)
    print values
    print [param.value for param in params]
