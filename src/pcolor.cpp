/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2010-2011 CEA
  Ludovic Aubry
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)
*/
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PyScalerArray
#include <numpy/arrayobject.h>
#ifdef _MSC_VER
    #include <float.h>
    #pragma fenv_access (on)
#else
    #include <fenv.h>
#endif
#include <math.h>
#ifdef _MSC_VER
    #define isnan(x) _isnan(x)
#endif
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "arrays.hpp"
#include "scaler.hpp"

using std::vector;
using std::min;
using std::max;
using std::swap;

#if 0
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
#endif

static bool vert_line(double _x0, double _y0, double _x1, double _y1, int NX,
		      vector<int>& imin, vector<int>& imax,
		      bool draw, npy_uint32 col, Array2D<npy_uint32>& D)
{
    int x0 = lrint(_x0);
    int y0 = lrint(_y0);
    int x1 = lrint(_x1);
    int y1 = lrint(_y1);
    int dx = abs(x1-x0);
    int dy = abs(y1-y0);
    int sx, sy;
    int NY=imin.size()-1;
    int err, e2;
    bool visible=false;
    NX = NX-1;
    if (x0 < x1)
	sx = 1;
    else
	sx = -1;
    if (y0 < y1)
	sy = 1;
    else
	sy = -1;
    err = dx-dy;

    do {
	if (y0>=0 && y0<=NY) {
	    int _min = min(imin[y0],x0);
	    int _max = max(imax[y0],x0);
	    if (draw) {
		if (x0>=0 && x0<=NX) {
		    D.value(x0,y0) = col;
		}
	    }
	    imin[y0] = max( 0,_min);
	    imax[y0] = min(NX,_max);
	    if (_min<=NX && _max>=0) {
		visible=true;
	    }
	}
	if ((x0 == x1) && (y0 == y1))
	    break;
	e2 = 2*err;
	if (e2 > -dy) {
	    err = err - dy;
	    x0 = x0 + sx;
	}
	if (e2 <  dx) {
	    err = err + dx;
	    y0 = y0 + sy;
	}
    } while(true);
    return visible;
}


template<class T>
struct QuadHelper {
    const Array2D<T>& X;
    const Array2D<T>& Y;
    const Array2D<T>& Z;
    Array2D<npy_uint32>& D;
    LutScale<T,npy_uint32>& scale;
    double x1, x2, y1, y2, m_dx, m_dy;
    npy_uint32 bgcolor;
    bool border;
    bool flat;
    double uflat, vflat;
    int ixmin, ixmax, iymin, iymax;

    QuadHelper( const Array2D<T>& X_,
		const Array2D<T>& Y_,
		const Array2D<T>& Z_,
		Array2D<npy_uint32>& D_,
		LutScale<T,npy_uint32>& scale_,
		double x1_, double x2_, double y1_, double y2_,
		bool _border, bool _flat,
		double _uflat, double _vflat
	):X(X_), Y(Y_), Z(Z_), D(D_), scale(scale_),
	  x1(x1_), x2(x2_), y1(y1_), y2(y2_),
	  bgcolor(0xff000000),
	  border(_border),
	  flat(_flat),uflat(_uflat),vflat(_vflat)
	{
	    m_dx = D.nj/(x2-x1);
	    m_dy = D.ni/(y2-y1);
	}

    void draw_triangles() {
	int i, j;
	vector<int> imin, imax;
	imin.resize(D.ni);
	imax.resize(D.ni);
	ixmin = D.nj;
	iymin = D.ni;
	ixmax = -1;
	iymax = -1;
	for(i=0;i<X.ni-1;++i) {
	    for(j=0;j<X.nj-1;++j) {
		draw_quad(i,j,imin,imax);
	    }
	}
    }

    void draw_quad(int qi, int qj,
		   vector<int>& imin, vector<int>& imax
	) {
	int i,j;
	double u, v;
	double v0, v1, v2, v3, v4;
	// Coordonnees du quad dans l'offscreen
	double ax = (X.value(qj+0,qi+0)-x1)*m_dx, ay=(Y.value(qj+0,qi+0)-y1)*m_dy;
	double bx = (X.value(qj+0,qi+1)-x1)*m_dx, by=(Y.value(qj+0,qi+1)-y1)*m_dy;
	double cx = (X.value(qj+1,qi+1)-x1)*m_dx, cy=(Y.value(qj+1,qi+1)-y1)*m_dy;
	double dx = (X.value(qj+1,qi+0)-x1)*m_dx, dy=(Y.value(qj+1,qi+0)-y1)*m_dy;
	// indice des sommets (A,B,C,D)<->0,1,2,3<->(qi,qj),(qi+1,qj),(qi+1,qj+1),(qi,qj+1)
	// trie par ordre x croissant ou y croissant (selon xarg, yarg)
	double ymin = min(ay,min(by,min(cy,dy)));
	double ymax = max(ay,max(by,max(cy,dy)));

	int i0 = int(ymin+.5);
	int i1 = int(ymax+.5);
//	printf("Quads: i=%d->%d\n", i0, i1);

	if (i0<0) i0=0;
	if (i1>=D.ni) i1=D.ni-1;
	if (i1<i0) return;

	iymin = min(iymin,i0);
	iymax = max(iymax,i1);
	for(i=i0;i<=i1;++i) {
	    imax[i]=-1;
	    imin[i]=D.nj;
	}

	// Compute the rasterized border of the quad
	bool visible = false;
	visible |= vert_line(ax,ay,bx,by,D.nj,imin,imax, border, 0xff000000, D);
	visible |= vert_line(bx,by,cx,cy,D.nj,imin,imax, border, 0xff000000, D);
	visible |= vert_line(cx,cy,dx,dy,D.nj,imin,imax, border, 0xff000000, D);
	visible |= vert_line(dx,dy,ax,ay,D.nj,imin,imax, border, 0xff000000, D);
	if (!visible)
	    return;

	double ex = ax+cx-dx-bx;
	double ey = ay+cy-dy-by;
	double n = 1./sqrt((cx-ax)*(cx-ax)+(cy-ay)*(cy-ay));
	if (n>1e2) n = 1.0;

	// Normalize vectors with ||AC||
	ax *= n; ay *= n;
	bx = bx*n-ax; by = by*n-ay;
	cx = cx*n-ax; cy = cy*n-ay;
	dx = dx*n-ax; dy = dy*n-ay;
	ex *= n; ey *= n;

	v1 = Z.value(qj,qi);
	v2 = Z.value(qj+1,qi);
	v3 = Z.value(qj+1,qi+1);
	v4 = Z.value(qj,qi+1);

	if (isnan(v1) || isnan(v2) || isnan(v3) || isnan(v4)) {
	    // XXX Color = Alpha
	    return ;
	}
	int dm=0, dM=0;
	if (border) {
	    dm=1;dM=-1;
	}
	npy_uint32 col = scale.eval( v1*(1-vflat)*(1-uflat) +
				 v2*  vflat  *(1-uflat) +
				 v3*  vflat  *  uflat   +
				 v4*(1-vflat)*  uflat   );
	for(i=i0+dm;i<=i1+dM;++i) {
	    ixmin = min(ixmin,imin[i]);
	    ixmax = max(ixmax,imax[i]);
	    int jmin=max(0,imin[i])+dm;
	    int jmax=min(imax[i],D.nj-1)+dM;
	    for(j=jmin;j<=jmax;++j) {
		if (!flat) {
		    params(j*n,i*n, ax,ay, bx,by, cx,cy, dx,dy, ex,ey, u,v);
		    if (u<0) u=0.; else if (u>1.) u=1.;
		    if (v<0) v=0.; else if (v>1.) v=1.;
		    /* v0 = v1*(1-v)*(1-u) + v2*v*(1-u) + v3*v*u + v4*(1-v)*u; */
		    v0 = u*( v*(v1-v2+v3-v4)+v4-v1 ) + v*(v2-v1) + v1;
		    col = scale.eval(v0);
		}
		D.value(j,i) = col;
	    }
	}
    }

    void params(double x, double y,
		double ax, double ay,
		double bx, double by,
		double cx, double cy,
		double dx, double dy,
		double ex, double ey,
		double& u, double&v) {
	/* solves AM=u.AB+v.AD+uv.AE with A,B,C,D quad, AE=DC+BA
	   M = (x,y)
	   with u^2.(AB^AE) +u.(AB^AD+AE^AM)+AD^AM=0
	   v = (AM-u.AB)/(AD+u.AE)
	*/

	double mx = x-ax, my = y-ay;
	double a1, a2, b, c, delta;

	if (false && (ex*ex+ey*ey)<1e-8) {
	    // fast case : parallelogram
	    if (fabs(dy)>1e-16) {
		double a=dx/dy;
		u = (mx-a*y)/(bx-a*by);
		v = (my-u*by)/dy;
		return ;
	    } else {
		double a=dy/dx;
		u = (my-a*x)/(by-a*bx);
		v = (mx-u*bx)/dx;
		return ;
	    }
	}
	a1 = bx*ey-ex*by;
	a2 = dx*ey-ex*dy;
	if (a1>a2) {
	    b = bx*dy-dx*by + ex*my-mx*ey;
	    c = dx*my-mx*dy;
	    if (fabs(a1)>1e-8) {
		delta = b*b-4*a1*c;
		u = (-b+sqrt(delta))/(2*a1);
	    } else {
		u = -c/b;
	    }
	    double den = (dx+u*ex);
	    if (fabs(den)>1e-8) {
		v = (mx - u*bx)/den;
	    } else {
		den = (dy+u*ey);
		v = (my - u*by)/den;
	    }
	} else {
	    b = dx*by-bx*dy + ex*my-mx*ey;
	    c = bx*my-mx*by;
	    if (fabs(a2)>1e-8) {
		delta = b*b-4*a2*c;
		v = (-b+sqrt(delta))/(2*a2);
	    } else {
		v = -c/b;
	    }
	    double den = (bx+v*ex);
	    if (fabs(den)>1e-8) {
		u = (mx - v*dx)/den;
	    } else {
		den = (by+v*ey);
		u = (my - v*dy)/den;
	    }
	}
#if 0
	if (isnan(u)) {
	    printf("AM=(%g,%g)\n", mx, my);
	    printf("AB=(%g,%g)\n", bx, by);
	    printf("AC=(%g,%g)\n", cx, cy);
	    printf("AD=(%g,%g)\n", dx, dy);
	    printf("AE=(%g,%g)\n", ex, ey);
	    printf("a1=%g, a2=%g, b=%g, c=%g\n", a1, a2, b, c);
	    printf("u=%g v=%g\n", u, v);
	}
#endif
    }
};


/**
   Draw a structured grid composed of quads (xy[i,j],xy[i+1,j],xy[i+1,j+1],xy[i,j+1] )
*/
PyObject *py_scale_quads(PyObject *self, PyObject *args)
{
    PyArrayObject *p_src_x=0, *p_src_y=0, *p_src_z=0, *p_dst=0;
    PyObject *p_lut_data, *p_dst_data, *p_interp_data, *p_src_data;
    double x1,x2,y1,y2;
    int border=0, flat=0;
    double uflat=0.5;
    double vflat=0.5;

    if (!PyArg_ParseTuple(args, "OOOOOOOO|i:_scale_quads",
			  &p_src_x, &p_src_y, &p_src_z, &p_src_data,
			  &p_dst, &p_dst_data,
			  &p_lut_data, &p_interp_data,
			  &border)) {
	return NULL;
    }
    if (!PyArg_ParseTuple(p_interp_data, "i|dd", &flat,&uflat,&vflat)) {
	PyErr_SetString(PyExc_ValueError, "Interpolation should be a tuple (type[,uflat,vflat])");
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
    #if PY_MAJOR_VERSION >= 3
        bg=PyLong_AsUnsignedLongMask(p_bg);
    #else
        bg=PyInt_AsUnsignedLongMask(p_bg);
    #endif
	if (PyErr_Occurred()) return NULL;
    }
    if (!check_lut(p_cmap)) {
	return NULL;
    }
    Array1D<npy_uint32> cmap(p_cmap);
    LutScale<npy_float64,npy_uint32>  scale(a, b, cmap, bg, apply_bg);
    QuadHelper<double> quad(X,Y,Z,dest,scale, x1, x2, y1, y2, border, flat, uflat, vflat);

    quad.draw_triangles();

    // examine source type
    return Py_BuildValue("iiii", quad.ixmin, quad.iymin, quad.ixmax, quad.iymax);
}

PyObject *py_vert_line(PyObject *self, PyObject *args)
{
    double x0,y0,x1,y1;
    int xmax;
    PyArrayObject *p_min, *p_max;

    if (!PyArg_ParseTuple(args, "ddddiOO:_vert_line", &x0,&y0,&x1,&y1,&xmax,&p_min,&p_max)) {
	return NULL;
    }
    if (!PyArray_Check(p_min) ||
	!PyArray_Check(p_max)) {
	PyErr_SetString(PyExc_TypeError, "imin, imax must be ndarray");
	return NULL;
    }
    if (PyArray_TYPE(p_min) != NPY_INT ||
	PyArray_TYPE(p_max) != NPY_INT) {
	PyErr_SetString(PyExc_TypeError, "imin, imax must be int ndarray");
	return NULL;
    }
    Array1D<int> pmin(p_min), pmax(p_max);
    vector<int> imin, imax;
    int nx = int(max(y0,y1))+1;
    if (pmin.ni<nx || pmax.ni<nx) {
	PyErr_SetString(PyExc_TypeError, "imin, imax not large enough");
	return NULL;
    }
    if (y0<0 || y1<0) {
	PyErr_SetString(PyExc_ValueError, "y bounds must be positive");
    }
    imin.resize(nx);
    imax.resize(nx);
    for(int i=0;i<nx;++i) {
	imin[i] = pmin.value(i);
	imax[i] = pmax.value(i);
    }
    Array2D<npy_uint32> dummy;
    vert_line(x0, y0, x1, y1, xmax, imin, imax, false, 0, dummy);
    for(int i=0;i<nx;++i) {
	pmin.value(i) = imin[i];
	pmax.value(i) = imax[i];
    }
    Py_INCREF(Py_None);
    return Py_None;
}
