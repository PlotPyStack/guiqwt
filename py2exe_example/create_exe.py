#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:11:57 2013

Buiding instructions:
python setup.py py2exe

@author: pierre
"""

from guidata import disthelpers as dh

def create_exe():
    dist = dh.Distribution()
    dist.setup('example', '1.0', 'guiqwt app example', 'simpledialog.pyw')
    dist.add_modules('guidata', 'guiqwt')
    dist.build_cx_freeze()  # use `build_py2exe` to use py2exe instead

if __name__ == '__main__':
    create_exe()
    