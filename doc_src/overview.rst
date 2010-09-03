Overview
========

Based on PyQwt (plot widgets for PyQt4 graphical user interfaces) and on 
scientific modules NumPy and SciPy, ``guiqwt`` is a Python library for efficient 
2-D data plotting (curves, 1-D and 2-D histograms, images) and signal/image 
processing application development.

The most popular Python module for data plotting is currently ``matplotlib``, 
an open-source library providing a lot of plot types and an API (the ``pylab`` 
interface) which is very close to MATLAB's plotting interface.

``guiqwt`` plotting features are quite limited in terms of plot types compared 
to ``matplotlib``. However the currently implemented plot types are much more 
efficient.
For example, the ``guiqwt`` image showing function (``imshow``) do not make any 
copy of the displayed data, hence allowing to show images much larger than 
with its ``matplotlib``'s counterpart. In other terms, when showing a 30-MB 
image (16-bits unsigned integers for example) with ``guiqwt``, no additional 
memory is wasted to display the image (except for the offscreen image of course 
which depends on the window size) whereas ``matplotlib`` takes more than 600-MB 
of additional memory (the original array is duplicated four times using 64-bits 
float data types).

``guiqwt`` also provides the following features:

    ``guiqwt.pyplot``: equivalent to ``matplotlib``'s pyplot module (``pylab``)

    supported plot items:
        * curves, error bar curves and 1-D histograms
        * images (RGB images are not supported), images with non-linear x/y 
          scales, images with specified pixel size (e.g. loaded from DICOM 
          files), 2-D histograms, pseudo-color images (``pcolor``)
        * labels, curve plot legends
        * shapes: polygon, polylines, rectangle, circle, ellipse and segment
        * annotated shapes (shapes with labels showing position and dimensions):
          rectangle with center position and size, circle with center position 
          and diameter, ellipse with center position and diameters (these items 
          are very useful to measure things directly on displayed images)

    curves, images and shapes:
        * multiple object selection for moving objects or editing their 
          properties through automatically generated dialog boxes (``guidata``)
        * item list panel: move objects from foreground to background, 
          show/hide objects, remove objects, ...
        * customizable aspect ratio
        * a lot of ready-to-use tools: plot canvas export to image file, image 
          snapshot, image rectangular filter, etc.

    curves:
        * interval selection tools with labels showing results of computing on 
          selected area
        * curve fitting tool with automatic fit, manual fit with sliders, ...

    images:
        * contrast adjustment panel: select the LUT by moving a range selection 
          object on the image levels histogram, eliminate outliers, ...
        * X-axis and Y-axis cross-sections: support for multiple images,
          average cross-section tool on a rectangular area, ...
        * apply any affine transform to displayed images in real-time (rotation,
          magnification, translation, horizontal/vertical flip, ...)

    application development helpers:
        * ready-to-use curve and image plot widgets and dialog boxes
        * load/save graphical objects (curves, images, shapes)
        * a lot of test scripts which demonstrate ``guiqwt`` features

