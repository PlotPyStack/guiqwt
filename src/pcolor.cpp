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
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "arrays.hpp"
#include "scaler.hpp"

using std::vector;
using std::min;
using std::max;
using std::swap;

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

static void vert_line(double _x0, double _y0, double _x1, double _y1, int NX,
	       vector<int>& imin, vector<int>& imax)
{
    int x0 = (int)(_x0+.5);
    int y0 = (int)(_y0+.5);
    int x1 = (int)(_x1+.5);
    int y1 = (int)(_y1+.5);
    int dx = abs(x1-x0);
    int dy = abs(y1-y0);
    int sx, sy;
    int NY=imin.size()-1;
    int err, e2;
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
	    imin[y0] = max(0,min(imin[y0],x0));
	    imax[y0] = min(NX,max(imax[y0],x0));
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
}

#if 0
// Calcule l'abcisse min et max d'une ligne (x0,y0)->(x1,y1)
// les valeurs sont rangees dans imin et imax a partir de 0 (int(x))
static void vert_line2(double x0, double y0, double x1, double y1, int NX,
		vector<int>& imin, vector<int>& imax)
{
    double m1 = (y1-y0)/(x1-x0);
    int NY=imin.size()-1;
    NX = NX - 1;
    if (fabs(m1)<1e-8) {
	// ligne horizontale
	int j = (int)(y0);
	if (j<0 || j>NY) return;
	if (x0>x1) {
	    imin[j] = min(imin[j],max(0,int(x1)));
	    imax[j] = max(imax[j],min(NX,int(x0)));
	} else {
	    imin[j] = min(imin[j],max(0,int(x0)));
	    imax[j] = max(imax[j],min(NX,int(x1)));
	}
	return;
    }
    if (fabs(m1)<=1.0) {
	int i,i1;
	double y;
	if (x0>x1) {
	    // pente horizontale
	    i  = (int)(x1);
	    i1 = min(NX,(int)(x0));
	    y = y1;
	} else {
	    // pente horizontale
	    i  = (int)(x0);
	    i1 = min(NX, (int)(x1));
	    y = y0;
	}
	while(i<0) {
	    i+=1;
	    y+=m1;
	}
	while(i<=i1) {
	    int j = (int)(y);
	    if (j<0 || j>NY) { i+=1; y+=m1; continue; }
	    imin[j] = min(imin[j],i);
	    imax[j] = max(imax[j],i);
	    i+=1;
	    y+=m1;
	}
    } else {
	double m2 = (x1-x0)/(y1-y0);
	if (fabs(m2)<1e-8) {
	    int j,j1;
	    // ligne verticale
	    int i = max(0,min(NX,(int)(x0)));
	    if (y0>y1) {
		j  = max(0,(int)(y1));
		j1 = min(NY,(int)(y0));
	    } else {
		j  = max(0,(int)(y0));
		j1 = min(NY,(int)(y1));
	    }
	    while(j<=j1) {
		imin[j] = min(imin[j],i);
		imax[j] = max(imax[j],i);
		j+=1;
	    }
	    return;
	}
	int j, j1;
	// pente verticale
	double x;
	if (y0>y1) {
	    j  = (int)(y1);
	    j1 = min(NY,(int)(y0));
	    x = x1;
	} else {
	    j  = (int)(y0);
	    j1 = min(NY,(int)(y1));
	    x = x0;
	}
	while(j<0) {
	    j+=1;
	    x+=m2;
	}
	while(j<=j1) {
	    int i = (int)(x);
	    imin[j] = min(imin[j],i);
	    imax[j] = max(imax[j],i);
	    j+=1;
	    x+=m2;
	}
    }
};
#endif

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
	    m_dx = (x2-x1)/D.nj;
	    m_dy = (y2-y1)/D.ni;
	}

    void draw(int i, int j,
	      int i1, int i2, int j1, int j2) {
	int k, l;
	double x, y, u, v;
	double v0, v1, v2, v3, v4;
	double ax = X.value(j+0,i+0), ay=Y.value(j+0,i+0);
	double bx = X.value(j+0,i+1)-ax, by=Y.value(j+0,i+1)-ay;
	double cx = X.value(j+1,i+1)-ax, cy=Y.value(j+1,i+1)-ay;
	double dx = X.value(j+1,i+0)-ax, dy=Y.value(j+1,i+0)-ay;
	double ex = cx-dx-bx, ey=cy-dy-by;
	double n = 1./sqrt(cx*cx+cy*cy);
	if (n>1e2) n = 1.0;
	// Normalize vectors with ||AC||
	ax *= n; ay *= n;
	bx *= n; by *= n;
	cx *= n; cy *= n;
	dx *= n; dy *= n;
	ex *= n; ey *= n;

	v1 = Z.value(j,i);
	v2 = Z.value(j+1,i);
	v3 = Z.value(j+1,i+1);
	v4 = Z.value(j,i+1);

	if (isnan(v1) || isnan(v2) || isnan(v3) || isnan(v4)) {
	    return ;
	}
	for(l=i1;l<i2;l+=1) {
	    for(k=j1;k<j2;k+=1) {
		x = x1+k*m_dx;
		y = y1+l*m_dy;
		params(x*n,y*n, ax,ay, bx,by, cx,cy, dx,dy, ex,ey, u,v);
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

    void draw_triangles(int &ix1, int &iy1, int&ix2, int&iy2, bool border) {
	int i, j;
	vector<int> imin, imax;
	imin.resize(D.ni);
	imax.resize(D.ni);
	for(i=0;i<X.ni-1;++i) {
	    for(j=0;j<X.nj-1;++j) {
		draw_quad(i,j,imin,imax,ix1,ix2,iy1,iy2,border);
	    }
	}
    }

    void draw_quad(int qi, int qj,
		   vector<int>& imin, vector<int>& imax,
		   int &ixmin, int &ixmax, int &iymin, int& iymax,
		   bool border
	) {
	int i,j;
	double u, v;
	double v0, v1, v2, v3, v4;
	// Coordonnees du quad dans l'offscreen
	double ax = (X.value(qj+0,qi+0)-x1)/m_dx, ay=(Y.value(qj+0,qi+0)-y1)/m_dy;
	double bx = (X.value(qj+0,qi+1)-x1)/m_dx, by=(Y.value(qj+0,qi+1)-y1)/m_dy;
	double cx = (X.value(qj+1,qi+1)-x1)/m_dx, cy=(Y.value(qj+1,qi+1)-y1)/m_dy;
	double dx = (X.value(qj+1,qi+0)-x1)/m_dx, dy=(Y.value(qj+1,qi+0)-y1)/m_dy;
	double ex = ax+cx-dx-bx, ey=ay+cy-dy-by;
	double n = 1./sqrt((cx-ax)*(cx-ax)+(cy-ay)*(cy-ay));
	// indice des sommets (A,B,C,D)<->0,1,2,3<->(qi,qj),(qi+1,qj),(qi+1,qj+1),(qi,qj+1)
	// trie par ordre x croissant ou y croissant (selon xarg, yarg)
	double ymin = min(double(D.ni),min(ay,min(by,min(cy,dy))));
	double ymax = max(0.,max(ay,max(by,max(cy,dy))));

	int i0 = max(0,int(ymin));
	int i1 = min(D.ni-1,int(ymax)+1);
//	printf("Quads: i=%d->%d\n", i0, i1);

	if (i0<0) i0=0;
	if (i1>=D.ni) i1=D.ni-1;
	if (i1<i0) return;

	iymin = min(iymin,i0);
	iymax = max(iymax,i1);
	for(i=i0;i<=i1;++i) {
	    imax[i]=0;
	    imin[i]=D.nj-1;
	}
	if (n>1e2) n = 1.0;

	// Compute the rasterized border of the quad
	vert_line(ax,ay,bx,by,D.nj,imin,imax);
	vert_line(bx,by,cx,cy,D.nj,imin,imax);
	vert_line(cx,cy,dx,dy,D.nj,imin,imax);
	vert_line(dx,dy,ax,ay,D.nj,imin,imax);

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
	for(i=i0;i<=i1;++i) {
//	    printf("%d: %d - %d\n", i, imin[i], imax[i]);
	    ixmin = min(ixmin,imin[i]);
	    ixmax = max(ixmax,imax[i]);
	    int jmin=max(0,imin[i]);
	    int jmax=min(imax[i],D.nj-1);
	    for(j=jmin;j<=jmax;++j) {
		params(j*n,i*n, ax,ay, bx,by, cx,cy, dx,dy, ex,ey, u,v);
		if (u<0) u=0.;
		else if (u>1) u=1.;
		if (v<0) v=0;
		else if (v>1) v=1.;
		v0 = v1*(1-v)*(1-u) +
		    v2*  v  *(1-u) +
		    v3*  v  *  u   +
		    v4*(1-v)*  u   ;
		D.value(j,i) = scale.eval(v0);
	    }
	    if (border && i<i1) {
		int jmin2=max(0,imin[i+1]);
		int jmax2=min(imax[i+1],D.nj-1);
		for (j=jmin2;j<=jmin;++j) {
		    D.value(j,i) = 0xff000000;
		}
		for (j=jmin;j<jmin2;++j) {
		    D.value(j,i) = 0xff000000;
		}
		for (j=jmax2;j<=jmax;++j) {
		    D.value(j,i) = 0xff000000;
		}
		for (j=jmax;j<jmax2;++j) {
		    D.value(j,i) = 0xff000000;
		}
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
	    if (den!=0.0) {
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
	    if (den!=0.0) {
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
    const Array2D<T>& X;
    const Array2D<T>& Y;
    const Array2D<T>& Z;
    Array2D<npy_uint32>& D;
    LutScale<T,npy_uint32>& scale;
    double x1, x2, y1, y2, m_dx, m_dy;
};

/**
   Draw a structured grid composed of quads (xy[i,j],xy[i+1,j],xy[i+1,j+1],xy[i,j+1] )
*/
PyObject *py_scale_quads(PyObject *self, PyObject *args)
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

PyObject *py_scale_quads2(PyObject *self, PyObject *args)
{
    PyArrayObject *p_src_x=0, *p_src_y=0, *p_src_z=0, *p_dst=0;
    PyObject *p_lut_data, *p_dst_data, *p_interp_data, *p_src_data;
    double x1,x2,y1,y2;
    int border=0;

    if (!PyArg_ParseTuple(args, "OOOOOOOO|i:_scale_quads",
			  &p_src_x, &p_src_y, &p_src_z, &p_src_data,
			  &p_dst, &p_dst_data,
			  &p_lut_data, &p_interp_data,
			  &border)) {
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
    int dni = PyArray_DIM(p_dst,0);
    int dnj = PyArray_DIM(p_dst,1);
    int dx1=dnj, dx2=-1, dy1=dni, dy2=-1;
    QuadHelper<double> quad(X,Y,Z,dest,scale, x1, x2, y1, y2);

    quad.draw_triangles(dx1,dy1,dx2,dy2, border);

    // examine source type
    return Py_BuildValue("iiii", dx1, dy1, dx2, dy2);
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
    vert_line(x0, y0, x1, y1, xmax, imin, imax);
    for(int i=0;i<nx;++i) {
	pmin.value(i) = imin[i];
	pmax.value(i) = imax[i];
    }
    Py_INCREF(Py_None);
    return Py_None;
}
