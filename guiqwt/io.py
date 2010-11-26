# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.io
---------

The `io` module provides input/output helper functions:
    * :py:func:`guiqwt.io.imagefile_to_array`: load an image (.png, .tiff, 
      .dicom, etc.) and return its data as a NumPy array
    * :py:func:`guiqwt.io.array_to_imagefile`: save an array to an image file
    * :py:func:`guiqwt.io.array_to_dicomfile`: save an array to a DICOM image 
      file according to a passed DICOM structure (base file)

Reference
~~~~~~~~~

.. autofunction:: imagefile_to_array
.. autofunction:: array_to_imagefile
.. autofunction:: array_to_dicomfile
"""

import sys, os.path as osp, numpy as np, os, time

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
          "I;16S": ('%si2' % _ENDIAN, None),
          "P": ('|u1', None),
          "RGB": ('|u1', 3),
          "RGBX": ('|u1', 4),
          "RGBA": ('|u1', 4),
          "CMYK": ('|u1', 4),
          "YCbCr": ('|u1', 4),
          }

# Image save mode constants
# they map directly to PIL modes
MODE_INTENSITY_S8  = "1"
MODE_INTENSITY_U8  = "L"
MODE_INTENSITY_S16 = "I;16S"
MODE_INTENSITY_U16 = "I;16"
MODE_INTENSITY_S32 = "I"
MODE_INTENSITY_FLOAT32 = "F"
MODE_RGB = "RGB"
MODE_RGBA = "RGBA"
VALID_MODES = [varname for varname in globals().keys()
               if varname.startswith("MODE_")]


def make_uid(root):
    uidparts = [root, str(time.time()),
                str(os.getpid()),
                str(np.random.randint(1000000)),
                ]
    return ".".join(uidparts)
                
def make_secondary_capture():
    from dicom.dataset import Dataset
    from dicom.UID import root
    ds = Dataset()
    uid = make_uid(root)
    ds.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    ds.MediaStorageSOPInstanceUID = uid
    ds.ImplementationClassUID = root+"2.2.2.2" # ???
    ds.ImplementationVersionName = "GUIQWT_10"
    ds.SpecificCharacterSet = "ISO_IR 192" # UTF-8
    ds.ImageType = ['DERIVED', 'SECONDARY']
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    ds.SOPInstanceUID = uid
    ds.preamble = "\x00"*128
    ds.isExplicitVR = True
    ds.isLittleEndian = True
    return ds

    
def set_dynamic_range_from_dtype(data, dtype):
    """WARNING modifies data in place"""
    info = np.iinfo(dtype)
    dmin = data.min()
    dmax = data.max()
    data -= dmin
    data *= float(info.max-info.min)/(dmax-dmin)
    data += float(info.min)
    return np.array(data, dtype)
    
def set_dynamic_range_from_mode(data, mode):
    """
    Set image dynamic range
    Mode: PIL modes (MODE_INTENSITY_U8, MODE_INTENSITY_U16, ...)
    
    *** WARNING modifies data in place ***
    """
    dtypes = {
              MODE_INTENSITY_U8: np.uint8,
              MODE_INTENSITY_S8: np.int8,
              MODE_INTENSITY_U16: np.uint16,
              MODE_INTENSITY_S16: np.int16,
             }
    return set_dynamic_range_from_dtype(data, dtypes[mode])


def imagefile_to_array(filename):
    """Return a numpy array from an image file *filename*"""
    if not isinstance(filename, basestring):
        filename = unicode(filename) # in case *filename* is a QString instance
    _base, ext = osp.splitext(filename)
    if ext.lower() in (".jpg", ".png", ".gif", ".tif"):
        import PIL.Image
        import PIL.TiffImagePlugin # py2exe
        img = PIL.Image.open(filename)
        if img.mode in ("RGB", "RGBX", "RGBA", "CMYK", "YCbCr"):
            # Converting RGB to greyscale
            img = img.convert("L")
        try:
            dtype, extra = DTYPES[img.mode]
        except KeyError:
            raise RuntimeError("%s mode is not supported" % img.mode)
        shape = (img.size[1], img.size[0])
        if extra is not None:
            shape += (extra,)
        arr = np.array(img.getdata(), dtype=np.dtype(dtype)).reshape(shape)
    elif ext.lower() in (".dcm",):
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
    else:
        raise RuntimeError("%s: unsupported image file"
                           % osp.basename(filename))
    # Converting RGB to greyscale
    if arr.ndim == 3:
        return arr.mean(axis=2)
    else:
        return arr

def array_to_imagefile(arr, filename, mode=None, max_range=False):
    """
    Save a numpy array *arr* into an image file *filename*
    Warning: option 'max_range' changes data in place
    """
    if max_range:
        arr = set_dynamic_range_from_mode(arr, mode)
    _base, ext = osp.splitext(filename)
    if ext.lower() in (".jpg", ".png", ".gif", ".tif"):
        import PIL.Image
        import PIL.TiffImagePlugin # py2exe
        if mode is None:
            for mode, (dtype, _extra) in DTYPES.iteritems():
                if dtype == arr.dtype.str:
                    break
            else:
                raise RuntimeError("Cannot determine PIL data type")
        img = PIL.Image.fromarray(arr, mode)
        img.save(filename)
    else:
        raise RuntimeError("%s: unsupported image file type" % ext)

def array_to_dicomfile(arr, dcmstruct, filename):
    """
    Save a numpy array *arr* into a DICOM image file *filename*
    based on DICOM structure *dcmstruct*
    """
#    import dicom
#    uid = make_uid(dicom.UID.root+"1")
    rows, columns = arr.shape
    infos = np.iinfo(arr.dtype)
    dcmstruct.BitsAllocated = infos.bits
    dcmstruct.BitsStored = infos.bits
    dcmstruct.HighBit = infos.bits-1
    dcmstruct.PixelRepresentation = ('u', 'i').index(infos.kind)
    dcmstruct.Rows = arr.shape[0]
    dcmstruct.Columns = arr.shape[1]
    dcmstruct.PhotometricInterpretation = 'MONOCHROME1'
#    print "LENGTH=", len(arr.tostring())
    dcmstruct.PixelData = arr.tostring()
    dcmstruct.save_as(filename)
    