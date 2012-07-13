# -*- coding: utf-8 -*-
#
# Copyright © 2009-2012 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

# pylint: disable=C0103

"""
guiqwt.io
---------

The `io` module provides input/output helper functions:
    * :py:func:`guiqwt.io.imread`: load an image (.png, .tiff, 
      .dicom, etc.) and return its data as a NumPy array
    * :py:func:`guiqwt.io.imwrite`: save an array to an image file

Reference
~~~~~~~~~

.. autofunction:: imread
.. autofunction:: imwrite
"""

#TODO: Implement an XML-based serialize/deserialize mechanism for plot items

import sys
import os.path as osp
import numpy as np

# Local imports
from guiqwt.config import _

    
def scale_data_to_dtype(data, dtype):
    """Scale array `data` to fit datatype `dtype` dynamic range
    
    WARNING: modifies data in place"""
    info = np.iinfo(dtype)
    dmin = data.min()
    dmax = data.max()
    data -= dmin
    data *= float(info.max-info.min)/(dmax-dmin)
    data += float(info.min)
    return np.array(data, dtype)
        
def eliminate_outliers(data, percent=2., bins=256):
    """Eliminate data histogram outliers"""
    hist, bin_edges = np.histogram(data, bins)
    from guiqwt.histogram import hist_range_threshold
    vmin, vmax = hist_range_threshold(hist, bin_edges, percent)
    return data.clip(vmin, vmax)


#===============================================================================
# I/O File type definitions
#===============================================================================
class FileType(object):
    """Filetype object:
        * `name` : description of filetype,
        * `read_func`, `write_func` : I/O callbacks,
        * `extensions` : filename extensions (with a dot!) or filenames,
        (list, tuple or space-separated string)"""        
    def __init__(self, name, extensions, read_func=None, write_func=None):
        self.name = name
        if isinstance(extensions, basestring):
            extensions = extensions.split()
        self.extensions = [osp.splitext(' '+ext)[1] for ext in extensions]
        self.read_func = read_func
        self.write_func = write_func
    
    @property
    def wcards(self):
        return "*"+(" *".join(self.extensions))
    
    @property
    def filters(self):
        return '\n%s (%s)' % (self.name, self.wcards)

class ImageIOHandler(object):
    """I/O handler: regroup all FileType objects"""
    def __init__(self):
        self.filetypes = []
        self._load_filters = ''
        self._save_filters = ''
    
    @property
    def allfilters(self):
        wcards = ' '.join([ftype.wcards for ftype in self.filetypes])
        return '%s (%s)' % (_("All supported files"), wcards)
    
    @property
    def load_filters(self):
        return self.allfilters + self._load_filters
    
    @property
    def save_filters(self):
        return self.allfilters + self._save_filters
    
    def add(self, name, extensions, read_func=None, write_func=None,
            import_func=None):
        if import_func is not None:
            try:
                import_func()
            except ImportError:
                return
        assert read_func is not None or write_func is not None
        ftype = FileType(name, extensions,
                         read_func=read_func, write_func=write_func)
        self.filetypes.append(ftype)
        if read_func:
            self._load_filters += ftype.filters
        if write_func:
            self._save_filters += ftype.filters
    
    def _get_filetype(self, ext):
        """Return FileType object associated to file extension `ext`"""
        for ftype in self.filetypes:
            if ext.lower() in ftype.extensions:
                return ftype
        else:
            raise RuntimeError("Unsupported file type: '%s'" % ext)
    
    def get_readfunc(self, ext):
        """Return read function associated to file extension `ext`"""
        ftype = self._get_filetype(ext)
        if ftype.read_func is None:
            raise RuntimeError("Unsupported file type (read): '%s'" % ext)
        else:
            return ftype.read_func
    
    def get_writefunc(self, ext):
        """Return read function associated to file extension `ext`"""
        ftype = self._get_filetype(ext)
        if ftype.write_func is None:
            raise RuntimeError("Unsupported file type (write): '%s'" % ext)
        else:
            return ftype.write_func

iohandler = ImageIOHandler()


#==============================================================================
# PIL-based Private I/O functions
#==============================================================================
if sys.byteorder == 'little':
    _ENDIAN = '<'
else:
    _ENDIAN = '>'

DTYPES = {
          "1": ('|b1', None),
          "L": ('|u1', None),
          "I": ('%si4' % _ENDIAN, None),
          "F": ('%sf4' % _ENDIAN, None),
          "I;16": ('%su2' % _ENDIAN, None),
          "I;16B": ('%su2' % _ENDIAN, None),
          "I;16S": ('%si2' % _ENDIAN, None),
          "P": ('|u1', None),
          "RGB": ('|u1', 3),
          "RGBX": ('|u1', 4),
          "RGBA": ('|u1', 4),
          "CMYK": ('|u1', 4),
          "YCbCr": ('|u1', 4),
          }

def _imread_pil(filename, to_grayscale=False):
    """Open image with PIL and return a NumPy array"""
    import PIL.Image
    import PIL.TiffImagePlugin # py2exe
    PIL.TiffImagePlugin.OPEN_INFO[(PIL.TiffImagePlugin.II,
                                   0, 1, 1, (16,), ())] = ("I;16", "I;16")
    img = PIL.Image.open(filename)
    if img.mode in ("CMYK", "YCbCr"):
        # Converting to RGB
        img = img.convert("RGB")
    if to_grayscale and img.mode in ("RGB", "RGBA", "RGBX"):
        # Converting to grayscale
        img = img.convert("L")
    try:
        dtype, extra = DTYPES[img.mode]
    except KeyError:
        raise RuntimeError("%s mode is not supported" % img.mode)
    shape = (img.size[1], img.size[0])
    if extra is not None:
        shape += (extra,)
    arr = np.array(img.getdata(), dtype=np.dtype(dtype)).reshape(shape)
    if img.mode in ("RGB", "RGBA", "RGBX"):
        arr = np.flipud(arr)
    return arr

def _imwrite_pil(filename, arr):
    """Write `arr` NumPy array to `filename` using PIL"""
    import PIL.Image
    import PIL.TiffImagePlugin # py2exe
    for mode, (dtype_str, _extra) in DTYPES.iteritems():
        if dtype_str == arr.dtype.str:
            break
    else:
        raise RuntimeError("Cannot determine PIL data type")
    img = PIL.Image.fromarray(arr, mode)
    img.save(filename)


#==============================================================================
# DICOM Private I/O functions
#==============================================================================
def _import_dcm():
    """DICOM Import function (checking for required libraries):
    DICOM support requires library `pydicom`"""
    import logging
    logger = logging.getLogger("pydicom")
    logger.setLevel(logging.CRITICAL)
    import dicom  # analysis:ignore
    logger.setLevel(logging.WARNING)

def _imread_dcm(filename):
    """Open DICOM image with pydicom and return a NumPy array"""
    import dicom
    dcm = dicom.ReadFile(filename)
    # **********************************************************************
    # The following is necessary until pydicom numpy support is improved:
    # (after that, a simple: 'arr = dcm.PixelArray' will work the same)
    format_str = '%sint%s' % (('u', '')[dcm.PixelRepresentation],
                              dcm.BitsAllocated)
    try:
        dtype = np.dtype(format_str)
    except TypeError:
        raise TypeError("Data type not understood by NumPy: "
                        "PixelRepresentation=%d, BitsAllocated=%d" % (
                        dcm.PixelRepresentation, dcm.BitsAllocated))
    arr = np.fromstring(dcm.PixelData, dtype)
    try:
        # pydicom 0.9.3:
        dcm_is_little_endian = dcm.isLittleEndian
    except AttributeError:
        # pydicom 0.9.4:
        dcm_is_little_endian = dcm.is_little_endian
    if dcm_is_little_endian != (sys.byteorder == 'little'):
        arr.byteswap(True)
    if hasattr(dcm, 'NumberofFrames') and dcm.NumberofFrames > 1:
        if dcm.SamplesperPixel > 1:
            arr = arr.reshape(dcm.SamplesperPixel, dcm.NumberofFrames,
                              dcm.Rows, dcm.Columns)
        else:
            arr = arr.reshape(dcm.NumberofFrames, dcm.Rows, dcm.Columns)
    else:
        if dcm.SamplesperPixel > 1:
            if dcm.BitsAllocated == 8:
                arr = arr.reshape(dcm.SamplesperPixel, dcm.Rows, dcm.Columns)
            else:
                raise NotImplementedError("This code only handles "
                            "SamplesPerPixel > 1 if Bits Allocated = 8")
        else:
            arr = arr.reshape(dcm.Rows, dcm.Columns)
    # **********************************************************************
    return arr

def _imwrite_dcm(filename, arr, template=None):
    """Save a numpy array `arr` into a DICOM image file `filename`
    based on DICOM structure `template`"""
    # Note: due to IOHandler formalism, `template` has to be a keyword argument
    assert template is not None,\
           "The `template` keyword argument is required to save DICOM files\n"\
           "(that is the template DICOM structure object)"
    infos = np.iinfo(arr.dtype)
    template.BitsAllocated = infos.bits
    template.BitsStored = infos.bits
    template.HighBit = infos.bits-1
    template.PixelRepresentation = ('u', 'i').index(infos.kind)
    data_vr = ('US', 'SS')[template.PixelRepresentation]
    template.Rows = arr.shape[0]
    template.Columns = arr.shape[1]
    template.SmallestImagePixelValue = int(arr.min())
    template[0x00280106].VR = data_vr
    template.LargestImagePixelValue = int(arr.max())
    template[0x00280107].VR = data_vr
    if not template.PhotometricInterpretation.startswith('MONOCHROME'):
        template.PhotometricInterpretation = 'MONOCHROME1'
    template.PixelData = arr.tostring()
    template[0x7fe00010].VR = 'OB'
    template.save_as(filename)


#==============================================================================
# Text files Private I/O functions
#==============================================================================
def _imread_txt(filename):
    """Open text file image and return a NumPy array"""
    for delimiter in ('\t', ',', ' ', ';'):
        try:
            return np.loadtxt(filename, delimiter=delimiter)
        except ValueError:
            continue
    else:
        raise

def _imwrite_txt(filename, arr):
    """Write `arr` NumPy array to text file `filename`"""
    if arr.dtype in (np.int8, np.uint8, np.int16, np.uint16,
                     np.int32, np.uint32):
        fmt = '%d'
    else:
        fmt = '%.18e'
    ext = osp.splitext(filename)[1]
    if ext.lower() in (".txt", ".asc", ""):
        np.savetxt(filename, arr, fmt=fmt)
    elif ext.lower() == ".csv":
        np.savetxt(filename, arr, fmt=fmt, delimiter=',')


#==============================================================================
# Registering I/O functions
#==============================================================================
iohandler.add(_(u"Portable images"), '*.png *.tif *.tiff',
              read_func=_imread_pil, write_func=_imwrite_pil)
iohandler.add(_(u"Portable compressed images"), '*.jpg *.gif',
              read_func=_imread_pil, write_func=_imwrite_pil)
iohandler.add(_(u"NumPy arrays"), '*.npy',
              read_func=np.load, write_func=np.save)
iohandler.add(_(u"Text files"), '*.txt *.csv *.asc',
              read_func=_imread_txt, write_func=_imwrite_txt)
iohandler.add(_(u"DICOM files"), '*.dcm', read_func=_imread_dcm,
              write_func=_imwrite_dcm, import_func=_import_dcm)


#==============================================================================
# Generic image read/write functions
#==============================================================================
def imread(fname, ext=None, to_grayscale=False):
    """Return a NumPy array from an image filename `fname`.
    
    If `to_grayscale` is True, convert RGB images to grayscale
    The `ext` (optional) argument is a string that specifies the file extension
    which defines the input format: when not specified, the input format is 
    guessed from filename."""
    if not isinstance(fname, basestring):
        fname = unicode(fname) # in case `filename` is a QString instance
    if ext is None:
        _base, ext = osp.splitext(fname)
    arr = iohandler.get_readfunc(ext)(fname)
    if to_grayscale and arr.ndim == 3:
        # Converting to grayscale
        return arr[...,:4].mean(axis=2)
    else:
        return arr

def imwrite(fname, arr, ext=None, dtype=None, max_range=None, **kwargs):
    """Save a NumPy array to an image filename `fname`.
    
    If `to_grayscale` is True, convert RGB images to grayscale
    The `ext` (optional) argument is a string that specifies the file extension
    which defines the input format: when not specified, the input format is 
    guessed from filename.
    If `max_range` is True, array data is scaled to fit the `dtype` (or data 
    type itself if `dtype` is None) dynamic range
    Warning: option `max_range` changes data in place"""
    if not isinstance(fname, basestring):
        fname = unicode(fname) # in case `filename` is a QString instance
    if ext is None:
        _base, ext = osp.splitext(fname)
    if max_range:
        arr = scale_data_to_dtype(arr, arr.dtype if dtype is None else dtype)
    iohandler.get_writefunc(ext)(fname, arr, **kwargs)


#==============================================================================
# Deprecated functions
#==============================================================================
def imagefile_to_array(filename, to_grayscale=False):
    """
    Return a NumPy array from an image file `filename`
    If `to_grayscale` is True, convert RGB images to grayscale
    """
    print >>sys.stderr,\
          "io.imagefile_to_array is deprecated: use io.imread instead"
    return imread(filename, to_grayscale=to_grayscale)

def array_to_imagefile(arr, filename, mode=None, max_range=False):
    """
    Save a numpy array `arr` into an image file `filename`
    Warning: option 'max_range' changes data in place
    """
    print >>sys.stderr,\
          "io.array_to_imagefile is deprecated: use io.imwrite instead"
    return imwrite(filename, arr, mode=mode, max_range=max_range)
    