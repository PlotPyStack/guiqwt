/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2009-2010 CEA
  Ludovic Aubry
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)
*/
#include <Python.h>
#include <numpy/arrayobject.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include "points.hpp"
#include "arrays.hpp"


enum {
    INTERP_NEAREST=0,
    INTERP_LINEAR=1,
    INTERP_AA=2
};

template <class T>
void swap(T& a, T&b)
{
    T x=a;
    a = b;
    b = x;
}

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

/* Scaler evaluates int(a*x+b) */
template<class T, bool is_int=num_trait<T>::is_integer>
struct Scaler {
    typedef num_trait<T> trait;

    Scaler(double _a, double _b):a(trait::fromdouble(_a)), b(trait::fromdouble(_b)) {}
    int scale(T x) const { return trait::toint(a*x+b); }
    typename trait::value_type a,b;
};
template<class T>
struct Scaler<T,true> {
    typedef num_trait<fixed> trait;

    Scaler(double _a, double _b):a(trait::fromdouble(_a)), b(trait::fromdouble(_b)) {}
    int scale(T x) const { return trait::toint(a*x+b); }
    typename trait::value_type a,b;
};

template<class T, class D>
class LinearScale {
public:
    typedef T source_type;
    typedef D dest_type;
    LinearScale(double _a, double _b, D _bg, bool apply_bg):s(_a, _b),bg(_bg), has_bg(apply_bg) {}

    D eval(T x) const {
	return s.scale(x);
    }
    void set_bg(D& dest) const {
	if (has_bg) dest = bg;
    }
protected:
    Scaler<T> s;
    D bg;
    bool has_bg;
};

template<class T, class D>
class LutScale
{
public:
    typedef T source_type;
    typedef D dest_type;
    LutScale(double _a, double _b, Array1D<D>& _lut,
	     D _bg, bool apply_bg):s(_a, _b),lut(_lut), bg(_bg), has_bg(apply_bg) {}

    D eval(T x) const {
	int val = s.scale(x);
	if (val<0) {
	    return lut.value(0);
	} else if (val>=lut.ni) {
	    return lut.value(lut.ni-1);
	}
	return lut.value(val);
    }
    void set_bg(D& dest) const {
	if (has_bg) dest = bg;
    }
protected:
    Scaler<T>  s;
    Array1D<D>& lut;
    D bg;
    bool has_bg;
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

	if (nx==0||nx==src.nj-1) return (T)v;
	if (ny==0||ny==src.ni-1) return (T)v;

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

	v0 = interp(src.value(nx-1,ny-1);
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
		it() = scale.eval(val);
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
	PyArray_TYPE(p_src) != NPY_UINT32 &&
	PyArray_TYPE(p_src) != NPY_INT32 &&
	PyArray_TYPE(p_src) != NPY_UINT16 &&
	PyArray_TYPE(p_src) != NPY_INT16 &&
	PyArray_TYPE(p_src) != NPY_UINT8 &&
	PyArray_TYPE(p_src) != NPY_INT8
	) {
	PyErr_Format(PyExc_TypeError,"%s data type must be one of the following:"
		     " double, float, uint32, int32, uint16, int16, uint8, int8", name);
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

static bool check_arrays(PyArrayObject* p_src, PyArrayObject *p_dest)
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

static bool check_lut(PyArrayObject *p_lut)
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
static bool scale_src(Params& p)
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
	    bg=PyInt_AsUnsignedLongMask(p_bg);
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

template <class Params>
static PyObject* dispatch_source(Params& p)
{
    bool ok;
    int dni = PyArray_DIM(p.p_dst, 0);
    int dnj = PyArray_DIM(p.p_dst, 1);

    if (!PyArg_ParseTuple(p.p_dst_data,"iiii",
			  &p.dx1, &p.dy1, &p.dx2, &p.dy2)) {
	PyErr_SetString(PyExc_ValueError, "Invalid destination rectangle");
    }
    if (p.dx2<p.dx1) swap(p.dx1,p.dx2);
    if (p.dy2<p.dy1) swap(p.dy1,p.dy2);
    check_image_bounds(dni, dnj, p.dx1, p.dy1);
    check_image_bounds(dni, dnj, p.dx2, p.dy2);

    switch(PyArray_TYPE(p.p_src)) {
    case NPY_FLOAT32:
	ok = scale_src<Params,npy_float32>(p);
	break;
    case NPY_FLOAT64:
	ok = scale_src<Params,npy_float64>(p);
	break;
    case NPY_UINT32:
	ok = scale_src<Params,npy_uint32>(p);
	break;
    case NPY_INT32:
	ok = scale_src<Params,npy_int32>(p);
	break;
    case NPY_UINT16:
	ok = scale_src<Params,npy_uint16>(p);
	break;
    case NPY_INT16:
	ok = scale_src<Params,npy_int16>(p);
	break;
    case NPY_UINT8:
	ok = scale_src<Params,npy_uint8>(p);
	break;
    case NPY_INT8:
	ok = scale_src<Params,npy_int8>(p);
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


/** return min(max(a,b,c,d),bound) */
static int max4(int a, int b, int c, int d, int bound)
{
    int x, y, z;
    x = (a>b ? a : b);
    y = (c>d ? c : d);
    z = (x>y ? x : y);
    return (z>bound ? bound : z);
}

/** return max(min(a,b,c,d),bound) */
    static int min4(int a, int b, int c, int d, int bound)
{
    int x, y, z;
    x = (a<b ? a : b);
    y = (c<d ? c : d);
    z = (x<y ? x : y);
    return (z<bound ? bound : z);
}

template<class T>
struct QuadHelper {
    QuadHelper( const Array2D<T>& X_,
		const Array2D<T>& Y_,
		const Array2D<T>& Z_,
		Array2D<npy_uint32>& D_,
		LutScale<T,npy_uint32>& scale_,
		double x1_, double x2_, double y1_, double y2_
	):X(X_), Y(Y_), Z(Z_), D(D_), scale(scale_),
	  x1(x1_), x2(x2_), y1(y1_), y2(y2_)
	{
	    dx = (x2-x1)/D.nj;
	    dy = (y2-y1)/D.ni;
	}

    void draw(int i, int j,
	      int i1, int i2, int j1, int j2) {
	int k, l;
	double x, y, u, v;
	double v0, v1, v2, v3, v4;
	v1 = Z.value(j,i);
	v2 = Z.value(j+1,i);
	v3 = Z.value(j+1,i+1);
	v4 = Z.value(j,i+1);
	
	for(l=i1;l<i2;l+=1) {
	    for(k=j1;k<j2;k+=1) {
		x = x1+k*dx;
		y = y1+l*dy;
		params(x,y,i,j,u,v);
		if (u>=0 && u<=1 && v>=0 && v<=1) {
		    v0 = v1*(1-v)*(1-u) +
			 v2*  v  *(1-u) +
			 v3*  v  *  u   +
			 v4*(1-v)*  u   ;
		    D.value(k,l) = scale.eval(v0);
		}
	    }
	}
    }
    void params(double x, double y, int i, int j, double& u, double&v) {
     
	/* solves AM=u.AB+v.AD+uv.AE with A,B,C,D quad, AE=DC+BA
	   M = (x,y)
	   with u^2.(AB^AE) +u.(AB^AD+AE^AM)+AD^AM=0
	   v = (AM-u.AB)/(AD+u.AE)
	*/
	double ax = X.value(j+0,i+0), ay=Y.value(j+0,i+0);
	double mx = x-ax, my = y-ay;
	double bx = X.value(j+0,i+1)-ax, by=Y.value(j+0,i+1)-ay;
	double cx = X.value(j+1,i+1)-ax, cy=Y.value(j+1,i+1)-ay;
	double dx = X.value(j+1,i+0)-ax, dy=Y.value(j+1,i+0)-ay;
	double ex = cx-dx-bx, ey=cy-dy-by;
	double a, b, c, delta;
	a = bx*ey-ex*by;
	b = bx*dy-dx*by + ex*my-mx*ey;
	c = dx*my-mx*dy;

	if (fabs(a)>1e-8) {
	    delta = b*b-4*a*c;
	    u = (-b+sqrt(delta))/(2*a);
//	    if (u<0) // useless ?
//		u = (-b-sqrt(delta))/(2*a);
	} else {
	    u = -c/b;
	}
	double den = (dx+u*ex);
	if (den!=0.0) {
	    v = (mx - u*bx)/den;
	} else {
	    den = (dy+u*ey);
	    v = (my - u*by)/den;
	}
#if 0
	if (isnan(u)) {
	    printf("AM=(%f,%f)\n", mx, my);
	    printf("AB=(%f,%f)\n", bx, by);
	    printf("AC=(%f,%f)\n", cx, cy);
	    printf("AD=(%f,%f)\n", dx, dy);
	    printf("AE=(%f,%f)\n", ex, ey);
	    printf("a=%f, b=%f, c=%f\n", a, b, c);
	    printf("u=%f v=%f\n", u, v);
	}
#endif
    }
    const Array2D<T>& X;
    const Array2D<T>& Y;
    const Array2D<T>& Z;
    Array2D<npy_uint32>& D;
    LutScale<T,npy_uint32>& scale;
    double x1, x2, y1, y2, dx, dy;
};

/**
   Draw a structured grid composed of quads (xy[i,j],xy[i+1,j],xy[i+1,j+1],xy[i,j+1] )
*/
static PyObject *py_scale_quads(PyObject *self, PyObject *args)
{
    PyArrayObject *p_src_x=0, *p_src_y=0, *p_src_z=0, *p_dst=0;
    PyObject *p_lut_data, *p_dst_data, *p_interp_data, *p_src_data;
    double x1,x2,y1,y2;

    if (!PyArg_ParseTuple(args, "OOOOOOOO:_scale_quads",
			  &p_src_x, &p_src_y, &p_src_z, &p_src_data,
			  &p_dst, &p_dst_data,
			  &p_lut_data, &p_interp_data)) {
	return NULL;
    }
    if (!check_arrays(p_src_x, p_dst)) {
	return NULL;
    }
    if (!PyArg_ParseTuple(p_src_data, "dddd:_scale_quads",
			  &x1, &y1, &x2, &y2)) {
	return NULL;
    }
    if (PyArray_TYPE(p_src_x)!=NPY_FLOAT64 ||
	PyArray_TYPE(p_src_y)!=NPY_FLOAT64 ||
	PyArray_TYPE(p_src_z)!=NPY_FLOAT64) {
	PyErr_SetString(PyExc_TypeError, "Only support float X,Y,Z");
	return NULL;
    }
    if (PyArray_TYPE(p_dst)!=NPY_UINT32) {
	PyErr_SetString(PyExc_TypeError, "Only support RGB dest for now");
	return NULL;
    }

    double a=1.0, b=0.0;
    PyObject* p_bg;
    PyArrayObject* p_cmap;
    bool apply_bg=true;
    if (!PyArg_ParseTuple(p_lut_data, "ddO|O", &a, &b, &p_bg, &p_cmap)) {
	PyErr_SetString(PyExc_ValueError, "Can't interpret pixel transformation tuple");
	return NULL;
    }
    if (p_bg==Py_None) apply_bg=false;

    Array2D<double> X(p_src_x), Y(p_src_y), Z(p_src_z);
    /* Destination is RGB */
    unsigned long bg=0;
    Array2D<npy_uint32> dest(p_dst);
    if (apply_bg) {
	bg=PyInt_AsUnsignedLongMask(p_bg);
	if (PyErr_Occurred()) return NULL;
    }
    if (!check_lut(p_cmap)) {
	return NULL;
    }
    Array1D<npy_uint32> cmap(p_cmap);
    LutScale<npy_float64,npy_uint32>  scale(a, b, cmap, bg, apply_bg);
    int ni = PyArray_DIM(p_src_x, 0);
    int nj = PyArray_DIM(p_src_x, 1);
    int dni = PyArray_DIM(p_dst,0);
    int dnj = PyArray_DIM(p_dst,1);
    double dx = dnj/(x2-x1);
    double dy = dni/(y2-y1);
    int dx1=dnj, dx2=-1, dy1=dni, dy2=-1;
    int i,j, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y;
    int i1, j1, i2, j2;
    QuadHelper<double> quad(X,Y,Z,dest,scale, x1, x2, y1, y2);

    for(i=0;i<ni-1;++i) {
	p1x = (int)((X.value(0,i+0)-x1)*dx+.5);
	p1y = (int)((Y.value(0,i+0)-y1)*dy+.5);
	p2x = (int)((X.value(0,i+1)-x1)*dx+.5);
	p2y = (int)((Y.value(0,i+1)-y1)*dy+.5);
	for(j=0;j<nj-1;++j) {
	    p3x = (int)((X.value(j+1,i+1)-x1)*dx+.5);
	    p3y = (int)((Y.value(j+1,i+1)-y1)*dy+.5);
	    p4x = (int)((X.value(j+1,i+0)-x1)*dx+.5);
	    p4y = (int)((Y.value(j+1,i+0)-y1)*dy+.5);

	    j1 = min4(p1x-1, p2x-1, p3x-1, p4x-1, 0);
	    j2 = max4(p1x+1, p2x+1, p3x+1, p4x+1, dnj);
	    i1 = min4(p1y-1, p2y-1, p3y-1, p4y-1, 0);
	    i2 = max4(p1y+1, p2y+1, p3y+1, p4y+1, dni);

	    quad.draw(i,j,i1,i2,j1,j2);

	    if (j1<dx1) dx1=j1;
	    if (j2>dx2) dx2=j2;
	    if (i1<dy1) dy1=i1;
	    if (i2>dy2) dy2=i2;
	    p1x = p4x;
	    p1y = p4y;
	    p2x = p3x;
	    p2y = p3y;
	}
    }


    // examine source type
    return Py_BuildValue("iiii", dx1, dy1, dx2, dy2);
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
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
init_scaler()
{
	PyObject* m;
	m = Py_InitModule("_scaler", _meths);
	import_array();
	PyModule_AddIntConstant(m,"INTERP_NEAREST", INTERP_NEAREST);
	PyModule_AddIntConstant(m,"INTERP_LINEAR", INTERP_LINEAR);
	PyModule_AddIntConstant(m,"INTERP_AA", INTERP_AA);
}
