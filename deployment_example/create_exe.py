#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deployment example
==================

Deployment script using `guidata.disthelpers` (py2exe or cx_Freeze)
"""

from guidata import disthelpers as dh

def create_exe():
    dist = dh.Distribution()
    dist.setup('example', '1.0', 'guiqwt app example', 'simpledialog.pyw')
    dist.add_modules('guidata', 'guiqwt')
    dist.build_cx_freeze()  # use `build_py2exe` to use py2exe instead

if __name__ == '__main__':
    create_exe()
    