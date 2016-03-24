/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2009-2010 CEA
  Ludovic Aubry
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)
*/
#include <Python.h>
#undef NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyScalerArray
#include <numpy/arrayobject.h>
#ifdef _MSC_VER
    #include <float.h>
    #pragma fenv_access (on)
    #define FE_TOWARDZERO    _RC_CHOP
    #define fegetround() (_controlfp(0,0) & _MCW_RC)
    int fesetround(int r) {
         if((r & _MCW_RC) == r) {  /* check the supplied value is one of 
    those allowed */
             int result = _controlfp(r, _MCW_RC) & _MCW_RC;
             return !(r == result);
         }
         return !0;
    }
#else
    #include <fenv.h>
#endif
#include <math.h>
#if defined(_MSC_VER) || defined(__MINGW32__)
    #define isnan(x) _isnan(x)
#endif
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "points.hpp"
#include "arrays.hpp"
#include "scaler.hpp"

using std::vector;
using std::min;
using std::max;
using std::swap;

enum {
    INTERP_NEAREST=0,
    INTERP_LINEAR=1,
    INTERP_AA=2
};

typedef union {
    npy_uint32 v;
    npy_uint8  c[4];
} rgba_t;

typedef XYTransform<Array1D<double> > XYScale;

template <class Transform>
struct params {
    typedef Transform transform_type;
    params(PyArrayObject* _src,
	   PyArrayObject* _dst, PyObject* _dst_data,
	   PyObject* _lut, PyObject* _interp,
	   Transform& _trans):p_dst(_dst),p_dst_data(_dst_data),
			      p_src(_src),
			      p_lut(_lut), p_interpolation(_interp),
			      trans(_trans) {
    }
    // Source, dest, coordinate transformation
    PyArrayObject *p_dst;
    PyObject *p_dst_data;
    PyArrayObject *p_src;
    PyObject* p_lut; // Pixel value transformation tuple
    PyObject* p_interpolation;
    Transform& trans;

    int dx1, dx2, dy1, dy2;
};


template<class T, class TR>
struct NearestInterpolation {
    T operator()(const Array2D<T>& src, const TR& tr, const typename TR::point& p) {
	int nx = p.ix();
	int ny = p.iy();
	return src.value(nx, ny);
    }
};

template<class T, class TR>
struct LinearInterpolation {
    T operator()(const Array2D<T>& src, const TR& tr, const typename TR::point& p) {
	int nx = p.ix();
	int ny = p.iy();
	double v = src.value(nx, ny);
	double a=0;

      // The following couple of lines were commented out to avoid disabling 
      // the linear interpolation on image edges. Demonstrating the effect of 
      // this change is quite easy: just try to show a very small image 
      // (e.g. 10x10) with guiqwt.pyplot.imshow for example.
//	if (nx==0||nx==src.nj-1) return (T)v;
//	if (ny==0||ny==src.ni-1) return (T)v;

	if (nx<src.nj-1) {
	    a = p.x()-nx;
	    v = (1-a)*v+a*src.value(nx+1,ny);
	}
	if (ny>=src.ni-1) return (T)v;
	double v2 = src.value(nx,ny+1);
	double b = p.y()-ny;
	if (nx<src.nj-1) {
	    v2 = (1-a)*v2+a*src.value(nx+1,ny+1);
	}
	return (T)(v*(1-b)+b*v2);
    }
};

template<class TR>
struct LinearInterpolation<npy_uint32,TR> {
    npy_uint32 operator()(const Array2D<npy_uint32>& src, const TR& tr, const typename TR::point& p) {
	int k;
	int nx = p.ix();
	int ny = p.iy();
	rgba_t p1, p2, p3, p4, r;
	p1.v = src.value(nx, ny);
	float v[4], v2[4];
	double a=0;
	if (nx<src.nj-1) {
	    p2.v = src.value(nx+1,ny);
	    a = p.x()-nx;
	    for(k=0;k<4;++k) {
		v[k] = (1-a)*p1.c[k]+a*p2.c[k];
	    }
	} else {
	    for(k=0;k<4;++k) {
		v[k] = p1.c[k];
	    }
	}
	if (ny>=src.ni-1) {
	    for(k=0;k<4;++k) {
		r.c[k] = (npy_uint8)(v[k]);
	    }
	    return r.v;
	}
	p3.v = src.value(nx,ny+1);
	double b = p.y()-ny;
	if (nx<src.nj-1) {
	    p4.v = src.value(nx+1,ny+1);
	    for(k=0;k<4;++k) {
		v2[k] = (1-a)*p3.c[k]+a*p4.c[k];
	    }
	} else {
	    for(k=0;k<4;++k) {
		v2[k] = p3.c[k];
	    }
	}
	for(k=0;k<4;++k) {
	    float px = v[k]*(1-b)+b*v2[k];
	    if (px<0.0) px = 0.0;
	    if (px>255.0) px = 255.0;
	    r.c[k] = (npy_uint8)px;
	}
	return r.v;
    }
};

#if 0
template<class T, class TR>
struct QuadInterpolation {
    T operator()(const Array2D<T>& src, const TR& tr, const typename TR::point& p) {
	int nx = p.ix();
	int ny = p.iy();
	double v = src.value(nx, ny);
	double a=0;
	if (nx==0||nx==src.nj-1) return src.value(nx,ny);
	if (ny==0||ny==src.ni-1) return src.value(nx,ny);

	v0 = interp(src.value(nx-1,ny-1));
	if (nx<src.nj-1) {
	    a = p.x()-nx;
	    v = (1-a)*v+a*src.value(nx+1,ny);
	}
	if (ny>=src.ni-1) return v;
	double v2 = src.value(nx,ny+1);
	double b = p.y()-ny;
	if (nx<src.nj-1) {
	    v2 = (1-a)*v2+a*src.value(nx+1,ny+1);
	}
	return v*(1-b)+b*v2;
    }
};
#endif

template<class T>
struct LinearInterpolation<T,XYScale> {
    T operator()(const Array2D<T>& src, const XYScale& tr, const typename XYScale::point& p) {
	int nx = p.ix();
	int ny = p.iy();
	double v = src.value(nx, ny);
	double a=0;

	if (nx==0||nx==src.nj-1) return (T)v;
	if (ny==0||ny==src.ni-1) return (T)v;

	if (nx<src.nj-1) {
	    double x0 = tr.ax.value(nx);
	    double x1 = tr.ax.value(nx+1);
	    a = (p.x()-x0)/(x1-x0);
	    v = (1-a)*v+a*src.value(nx+1,ny);
	}
	if (ny>=src.ni-1) return (T)v;
	double v2 = src.value(nx,ny+1);
	double y0 = tr.ay.value(ny);
	double y1 = tr.ay.value(ny+1);
	double b = (p.y()-y0)/(y1-y0);
	if (nx<src.nj-1) {
	    v2 = (1-a)*v2+a*src.value(nx+1,ny+1);
	}
	return (T)(v*(1-b)+b*v2);
    }
};

template<>
struct LinearInterpolation<npy_uint32,XYScale> {
    npy_uint32 operator()(const Array2D<npy_uint32>& src, const XYScale& tr, const XYScale::point& p) {
	return 0;
    }
};

template<class T, class TR>
struct SubSampleInterpolation
{
    SubSampleInterpolation(const Array2D<T>& _mask):mask(_mask) {
	ki = 1./(mask.ni-1);
	kj = 1./(mask.nj-1);
    }
    T operator()(const Array2D<T>& src, const TR& tr, const typename TR::point& p0) {
	int i,j;
	typename TR::point p, p1;
	typename num_trait<T>::large_type value = 0;
	typename num_trait<T>::large_type count=0, msk, val;
	p1.copy(p0);
	tr.incy(p1,-0.5);
	tr.incx(p1,-0.5);
	for(i=0;i<mask.ni;++i) {
	    p.copy(p1);
	    for(j=0;j<mask.nj;++j) {
		if (p.inside()) {
		    msk = mask.value(j,i);
		    val = src.value(p.ix(), p.iy());
		    value += msk*val;
		    count += msk;
		    //printf("i,j=%d,%d : %f,%f / %d,%d = %lfx%lf\n", i, j, p.x(), p.y(), p.ix(), p.iy(), (double)val, (double)msk );
		}
		tr.incx(p,kj);
	    }
	    tr.incy(p1,ki);
	}
	//printf("%d/%d\n", (int)value, (int)count);
	if (count)
	    return value/count;
	else
	    return value;
    }
    typename TR::real ki, kj;
    const Array2D<T>& mask;
};

template<class DEST, class ST, class Scale, class Trans, class Interpolation>
void _scale_rgb(DEST& dest,
		Array2D<ST>& src, const Scale& scale, const Trans& tr,
		int dx1, int dy1, int dx2, int dy2,
		Interpolation& interpolate)
{
    int i, j;
    ST val;
    int round = fegetround();
    PixelIterator<DEST> it(dest);
    typename Trans::point p, p0;

    fesetround(FE_TOWARDZERO);
    /*
    printf("SRC: ni=%d nj=%d\n", src.ni, src.nj);
    printf("TR: dx=%lf dy=%lf\n", tr.dx, tr.dy);
    printf("DST: ni=%d nj=%d si=%d sj=%d\n", dest.ni, dest.nj, dest.si, dest.sj);
    */
    tr.set(p0, dx1, dy1);
    for(i=dy1;i<dy2;++i) {
	it.moveto(dx1, i);
	p = p0;
	for(j=dx1;j<dx2;++j) {
	    if (!p.inside()) {
		scale.set_bg( it() );
	    } else {
		val = interpolate(src, tr, p);
		if (isnan((float) val)) {
		    scale.set_bg( it() );
		} else {
		    it() = scale.eval(val);
		}
	    }
	    tr.incx(p);
	    it.move(1,0);
	}
	tr.incy(p0);
    }
    fesetround(round);
}

static bool check_dispatch_type(const char* name, PyArrayObject* p_src)
{
    if (PyArray_TYPE(p_src) != NPY_DOUBLE &&
	PyArray_TYPE(p_src) != NPY_FLOAT &&
	PyArray_TYPE(p_src) != NPY_UINT64 &&
	PyArray_TYPE(p_src) != NPY_INT64 &&
	PyArray_TYPE(p_src) != NPY_UINT32 &&
	PyArray_TYPE(p_src) != NPY_INT32 &&
	PyArray_TYPE(p_src) != NPY_UINT16 &&
	PyArray_TYPE(p_src) != NPY_INT16 &&
	PyArray_TYPE(p_src) != NPY_UINT8 &&
	PyArray_TYPE(p_src) != NPY_INT8 &&
	PyArray_TYPE(p_src) != NPY_BOOL
	) {
	PyErr_Format(PyExc_TypeError,"%s data type must be one of the following:"
           " double, float, uint64, int64, uint32, int32, uint16, int16, uint8, int8, bool", name);
	return false;
    }
    return true;
}

static bool check_array_2d(const char* name, PyArrayObject *arr, int dtype)
{
    if (!PyArray_Check(arr)) {
	PyErr_Format(PyExc_TypeError, "%s must be a ndarray", name);
	return false;
    }
    if (arr->nd!=2) {
	PyErr_Format(PyExc_TypeError,"%s must be 2-D array", name);
	return false;
    }
    if (dtype>=0 && PyArray_TYPE(arr) != dtype) {
	PyErr_Format(PyExc_TypeError,"%s data type must be %d", name, dtype);
	return false;
    }
    return true;
}

bool check_arrays(PyArrayObject* p_src, PyArrayObject *p_dest)
{
    if (!PyArray_Check(p_src) || !PyArray_Check(p_dest)) {
	PyErr_SetString(PyExc_TypeError,"src and dst must be ndarrays");
	return false;
    }
    if (PyArray_TYPE(p_dest) != NPY_UINT32 &&
	PyArray_TYPE(p_dest) != NPY_FLOAT32 &&
	PyArray_TYPE(p_dest) != NPY_FLOAT64) {
	PyErr_SetString(PyExc_TypeError,"dst data type must be uint32 or float");
	return false;
    }
    if (p_src->nd!=2 || p_dest->nd!=2) {
	PyErr_SetString(PyExc_TypeError,"dst and src must be 2-D arrays");
	return false;
    }
    return check_dispatch_type("src", p_src);
}

bool check_lut(PyArrayObject *p_lut)
{
    if (!PyArray_Check(p_lut)) {
	PyErr_SetString(PyExc_TypeError,"lut must be an ndarray");
	return false;
    }
    if (p_lut->nd!=1) {
	PyErr_SetString(PyExc_TypeError,"lut must be a 1D array");
	return false;
    }
    if (PyArray_TYPE(p_lut) != NPY_UINT32) {
	PyErr_SetString(PyExc_TypeError,"lut data type must be uint32");
	return false;
    }
    return true;
}

static bool check_transform(PyArrayObject* p_tr)
{
    if (!PyArray_Check(p_tr)) {
	PyErr_SetString(PyExc_TypeError,"transform must be an ndarray");
	return false;
    }
    if (PyArray_TYPE(p_tr) != NPY_DOUBLE) {
	PyErr_SetString(PyExc_TypeError,"transform data type must be float");
	return false;
    }
    int ni = PyArray_DIM(p_tr, 0);
    int nj = PyArray_DIM(p_tr, 1);
    
    if (ni!=3||nj!=3) {
	PyErr_SetString(PyExc_TypeError,"transform must be 3x3");
	return false;
    }
    return true;
}

static void check_image_bounds(int ni, int nj, int& dx, int &dy)
{
    if (dx<0) dx=0;
    if (dy<0) dy=0;
    if (dx>nj) dx=nj;
    if (dy>ni) dy=ni;
}

template<class Params, class PixelScale, class Interp>
static bool scale_src_dst_interp(Params& p, PixelScale& pixel_scale, Interp& interp)
{
    typedef typename PixelScale::source_type ST;
    typedef typename PixelScale::dest_type DT;
    typedef typename Params::transform_type Transform;

    Array2D<ST> src(p.p_src);
    Array2D<DT> dst(p.p_dst);
    
    _scale_rgb(dst, src, pixel_scale, p.trans,
	       p.dx1, p.dy1, p.dx2, p.dy2, interp);
    return true;
}

template <class Params, class PixelScale>
static bool scale_src_dst(Params& p, PixelScale& pixel_scale)
{
    typedef typename PixelScale::source_type ST;
    typedef typename Params::transform_type TR;
    typedef NearestInterpolation<ST, TR> Nearest;
    typedef SubSampleInterpolation<ST, TR> SubAA;
    typedef LinearInterpolation<ST, TR> Linear;

    int interpolation;
    PyArrayObject *p_mask=0;

    if (!PyArg_ParseTuple(p.p_interpolation, "i|O", &interpolation, &p_mask)) {
	PyErr_SetString(PyExc_ValueError, "Can't interpret interpolation");
	return false;
    }
    switch(interpolation) {
    case INTERP_NEAREST: {
	Nearest interp;
	return scale_src_dst_interp<Params, PixelScale, Nearest>(p, pixel_scale, interp);
    }
    case INTERP_AA: {
	if (!check_array_2d("AA Mask", p_mask, PyArray_TYPE(p.p_src))) return false;
	Array2D<ST> mask(p_mask);
	SubAA interp(mask);
	return scale_src_dst_interp<Params, PixelScale, SubAA>(p, pixel_scale, interp);
    }
    case INTERP_LINEAR: {
	Linear interp;
	return scale_src_dst_interp<Params, PixelScale, Linear>(p, pixel_scale, interp);
    }
    default:
	PyErr_SetString(PyExc_ValueError, "Unknown interpolation type");
	return false;
    };
}
/* we know the transformation and source type, now we dispatch
   on the destination type, which determines the LUT transformation
*/
template <class Params, class ST>
static bool scale_src_bw(Params& p)
{
    typedef LutScale<ST,npy_uint32> color_scale;
    typedef LinearScale<ST,npy_float32> bw32_scale;
    typedef LinearScale<ST,npy_float64> bw64_scale;
    double a, b;
    PyObject* p_bg;
    PyArrayObject *p_cmap=0;
    bool apply_bg=true;

    if (!PyArg_ParseTuple(p.p_lut, "ddO|O", &a, &b, &p_bg, &p_cmap)) {
	PyErr_SetString(PyExc_ValueError, "Can't interpret pixel transformation tuple");
	return false;
    }
    if (p_bg==Py_None) apply_bg=false;

    switch(PyArray_TYPE(p.p_dst)) {
    case NPY_UINT32: {
	/* Destination is RGB */
	unsigned long bg=0;
	if (apply_bg) {
        #if PY_MAJOR_VERSION >= 3
            bg=PyLong_AsUnsignedLongMask(p_bg);
        #else
            bg=PyInt_AsUnsignedLongMask(p_bg);
        #endif    
	    if (PyErr_Occurred()) return false;
	}
	if (!check_lut(p_cmap)) {
	    return false;
	}
	Array1D<npy_uint32> cmap(p_cmap);
	color_scale  scale(a, b, cmap, bg, apply_bg);
	return scale_src_dst<Params,color_scale>(p, scale);
    }
    case NPY_FLOAT32: {
	double bg=0.0;
	if (apply_bg) {
	    bg=PyFloat_AsDouble(p_bg);
	    if (PyErr_Occurred()) return false;
	}
	bw32_scale scale(a, b, bg, apply_bg);
	return scale_src_dst<Params,bw32_scale>(p, scale);
    }
    case NPY_FLOAT64: {
	double bg=0.0;
	if (apply_bg) {
	    bg=PyFloat_AsDouble(p_bg);
	    if (PyErr_Occurred()) return false;
	}
	bw64_scale scale(a, b, bg, apply_bg);
	return scale_src_dst<Params,bw64_scale>(p, scale);
    }
    default:
	PyErr_SetString(PyExc_TypeError,"Destination array must be uint32 (rgb) or float (BW)");
	return false;
    }
}

template <class Params, class ST>
static bool scale_src_rgb(Params& p)
{
    // Special case p_lut = None
    NoScale<ST,npy_uint32> scale(0, false);
    return scale_src_dst<Params,NoScale<ST,npy_uint32> >(p, scale);
}

template <class Params>
static PyObject* dispatch_source(Params& p)
{
    bool ok;
    int dni = PyArray_DIM(p.p_dst, 0);
    int dnj = PyArray_DIM(p.p_dst, 1);

    if (!PyArg_ParseTuple(p.p_dst_data,"iiii",
			  &p.dx1, &p.dy1, &p.dx2, &p.dy2)) {
	PyErr_SetString(PyExc_ValueError, "Invalid destination rectangle (expected tuple of 4 integers)");
	return NULL;
    }
    if (p.dx2<p.dx1) swap(p.dx1,p.dx2);
    if (p.dy2<p.dy1) swap(p.dy1,p.dy2);
    check_image_bounds(dni, dnj, p.dx1, p.dy1);
    check_image_bounds(dni, dnj, p.dx2, p.dy2);

    switch(PyArray_TYPE(p.p_src)) {
    case NPY_FLOAT32:
	ok = scale_src_bw<Params,npy_float32>(p);
	break;
    case NPY_FLOAT64:
	ok = scale_src_bw<Params,npy_float64>(p);
	break;
    case NPY_UINT64:
	ok = scale_src_bw<Params,npy_uint64>(p);
	break;
    case NPY_INT64:
	ok = scale_src_bw<Params,npy_int64>(p);
	break;
    case NPY_INT32:
	ok = scale_src_bw<Params,npy_int32>(p);
	break;
    case NPY_UINT32: // RGBA
	ok = scale_src_rgb<Params,npy_uint32>(p);
	break;
    case NPY_UINT16:
	ok = scale_src_bw<Params,npy_uint16>(p);
	break;
    case NPY_INT16:
	ok = scale_src_bw<Params,npy_int16>(p);
	break;
    case NPY_UINT8:
	ok = scale_src_bw<Params,npy_uint8>(p);
	break;
    case NPY_INT8:
	ok = scale_src_bw<Params,npy_int8>(p);
	break;
    case NPY_BOOL:
	ok = scale_src_bw<Params,npy_bool>(p);
	break;
    default:
	PyErr_SetString(PyExc_TypeError,"Unknown source data type");
	return NULL;
    }
    if (!ok)
	return NULL;
    return Py_BuildValue("iiii", p.dx1, p.dy1, p.dx2, p.dy2);
}

/* Input data :
   
   SRC, SRC_DATA, DST, DST_DATA, LUT_DATA

   SRC : PyArrayObject (i8,u8,i16,u16,float32,float64)
   DST : PyArrayObject : u32 -> rgb, float32 : bw
   SRC_DATA : varies :
       Scale : source rect (x1,y1,x2,y2)
       Transform : transformation matrix
       XY : source rect, X array, Y array
   DST_DATA : dest rect (dx1,dy1,dx2,dy2)
   LUT_DATA : (a,b,bg) if DST is bw or (a,b,bg,cmap) if DST is rgb
*/

static PyObject *py_scale_xy(PyObject *self, PyObject *args)
{
    typedef params<XYScale> Params;
    PyArrayObject *p_src=0, *p_dst=0, *p_ax=0, *p_ay=0;
    PyObject *p_lut_data, *p_src_data, *p_dst_data, *p_interp_data;
    double x1, y1, x2, y2;

    if (!PyArg_ParseTuple(args, "OOOOOO:_scale_xy",
			  &p_src, &p_src_data,
			  &p_dst, &p_dst_data,
			  &p_lut_data, &p_interp_data)) {
	return NULL;
    }
    if (!check_arrays(p_src, p_dst)) {
	return NULL;
    }
    if (!PyArg_ParseTuple(p_src_data, "OO(dddd):_scale_xy",
			  &p_ax, &p_ay, &x1, &y1, &x2, &y2)) {
	return NULL;
    }
    int ni = PyArray_DIM(p_src, 0);
    int nj = PyArray_DIM(p_src, 1);
    int dni = PyArray_DIM(p_dst, 0);
    int dnj = PyArray_DIM(p_dst, 1);
    double dx = (x2-x1)/dnj;
    double dy = (y2-y1)/dni;
    Array1D<double> ax(p_ax), ay(p_ay);
    XYScale trans(nj, ni, x1, y1, dx, dy, ax, ay);
    Params scale_params(p_src, p_dst, p_dst_data,
			p_lut_data, p_interp_data, trans);

    // examine source type
    return dispatch_source<Params>(scale_params);
}

static PyObject *py_scale_tr(PyObject *self, PyObject *args)
{
    typedef params<LinearTransform> Params;
    PyArrayObject *p_src=0, *p_dst=0, *p_tr;
    PyObject *p_lut_data, *p_dst_data, *p_interp_data;

    if (!PyArg_ParseTuple(args, "OOOOOO:_scale_tr",
			  &p_src, &p_tr,
			  &p_dst, &p_dst_data,
			  &p_lut_data, &p_interp_data)) {
	return NULL;
    }
    if (!check_arrays(p_src, p_dst)) {
	return NULL;
    }
    if (!check_transform(p_tr)) {
	    return NULL;
    }

    int ni = PyArray_DIM(p_src, 0);
    int nj = PyArray_DIM(p_src, 1);
    Array2D<double>  tr(p_tr);
    LinearTransform trans(nj, ni,
			  tr.value(2,0), tr.value(2,1),  // x0, y0
			  tr.value(0,0), tr.value(1,0),  // xx, xy
			  tr.value(0,1), tr.value(1,1)  // yx, yy
	);
    Params scale_params(p_src, p_dst, p_dst_data,
			p_lut_data, p_interp_data, trans);

    // examine source type
    return dispatch_source<Params>(scale_params);
}

static PyObject *py_scale_rect(PyObject *self, PyObject *args)
{
    typedef params<ScaleTransform> Params;
    PyArrayObject *p_src=0, *p_dst=0;
    PyObject *p_lut_data, *p_dst_data, *p_interp_data, *p_src_data;
    double x1,x2,y1,y2;

    if (!PyArg_ParseTuple(args, "OOOOOO:_scale_rect",
			  &p_src, &p_src_data,
			  &p_dst, &p_dst_data,
			  &p_lut_data, &p_interp_data)) {
	return NULL;
    }
    if (!check_arrays(p_src, p_dst)) {
	return NULL;
    }
    if (!PyArg_ParseTuple(p_src_data, "dddd:_scale_rect",
			  &x1, &y1, &x2, &y2)) {
	return NULL;
    }

    int ni = PyArray_DIM(p_src, 0);
    int nj = PyArray_DIM(p_src, 1);
    int dni = PyArray_DIM(p_dst,0);
    int dnj = PyArray_DIM(p_dst,1);
    double dx = (x2-x1)/dnj;
    double dy = (y2-y1)/dni;
    ScaleTransform trans(nj, ni, x1, y1, dx, dy );

    Params scale_params(p_src, p_dst, p_dst_data,
			p_lut_data, p_interp_data, trans);

    // examine source type
    return dispatch_source<Params>(scale_params);
}



class Histogram {
public:
    Histogram(PyArrayObject *_data, PyArrayObject *_bins,
	      PyArrayObject *_res, int ignore_bounds):p_data(_data),
						      p_bins(_bins),
						      p_res(_res),
						      mode(ignore_bounds) {
    }

    template<class T> void run() {
	Array1D<npy_uint32> res(p_res);
	Array1D<T> data(p_data);
	Array1D<T> bins(p_bins);
	typename Array1D<T>::iterator it, end, bin, bin0, bin1;
	end = data.end();
	bin0 = bins.begin();
	bin1 = bins.end();
	for(it=data.begin();it<end;++it) {
	    bin = std::lower_bound(bin0, bin1, *it);
	    res.value(bin-bin0)++;
	}
    }
    PyArrayObject *p_data, *p_bins, *p_res;
    int mode;
};

static PyObject *py_histogram(PyObject *self, PyObject *args)
{
    PyArrayObject *p_data=0, *p_bins=0, *p_res=0;
    int ignore_bounds=0;

    if (!PyArg_ParseTuple(args, "OOO|i:_histogram", &p_data, &p_bins, &p_res,
			  &ignore_bounds)) {
	return NULL;
    }
    if (!PyArray_Check(p_data) ||
	!PyArray_Check(p_bins) ||
	!PyArray_Check(p_res)) {
	PyErr_SetString(PyExc_TypeError, "data, bins, dest must be ndarray");
	return NULL;
    }
    if (!check_dispatch_type("data", p_data)) {
	return NULL;
    }
    Histogram hist(p_data,p_bins,p_res,ignore_bounds);
    dispatch_array(PyArray_TYPE(p_data), hist);
    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *py_vert_line(PyObject *self, PyObject *args);
PyObject *py_scale_quads(PyObject *self, PyObject *args);

static PyMethodDef _meths[] = {
    {"_scale_xy",  py_scale_xy, METH_VARARGS,
     "transform source to destination with arbitrary X and Y scale"},
    {"_scale_tr",  py_scale_tr, METH_VARARGS,
     "Linear transformation of source to destination"},
    {"_scale_rect",  py_scale_rect, METH_VARARGS,
     "Linear rescale of source to destination parallel to axes"},
    {"_scale_quads",  py_scale_quads, METH_VARARGS,
     "Linear rescale of a structured grid to destination parallel to axes"},
    {"_histogram", py_histogram, METH_VARARGS,
     "Compute histogram of 1d data"},
    {"_line_test", py_vert_line, METH_VARARGS,
     "Rasterize lines"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_scaler",           /* m_name */
        "Scaler module",     /* m_doc */
        -1,                  /* m_size */
        _meths,              /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__scaler(void)
#else
init_scaler()
#endif
{
	PyObject* m;
    #if PY_MAJOR_VERSION >= 3
        m = PyModule_Create(&moduledef);
    #else
        m = Py_InitModule("_scaler", _meths);
    #endif
	import_array();
	PyModule_AddIntConstant(m,"INTERP_NEAREST", INTERP_NEAREST);
	PyModule_AddIntConstant(m,"INTERP_LINEAR", INTERP_LINEAR);
	PyModule_AddIntConstant(m,"INTERP_AA", INTERP_AA);

    #if PY_MAJOR_VERSION >= 3
        return m;
    #endif
}
