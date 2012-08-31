/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2009-2010 CEA
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)
*/
#ifndef __POINTS_HPP__
#define __POINTS_HPP__

#include "traits.hpp"



/*
  This file defines several classes of 2D points
  which are used by other algorithms
*/

class Point {
public:
    typedef double real;
    typedef num_trait<real> trait;
    Point():_ix(0),_iy(0),_x(0.0),_y(0.0) {}
    real x() const { return _x; }
    real y() const { return _y; }
    int ix() const { return _ix; }
    int iy() const { return _iy; }

    void setx(real x) {
	_x = x;
	_ix = trait::toint(x);
    }
    void sety(real y) {
	_y = y;
	_iy = trait::toint(y);
    }
    void set(real x, real y) {
	setx(x);
	sety(y);
    }
    void copy(const Point& p) {
	_ix = p._ix; _iy = p._iy;
	_x = p._x; _y = p._y;
    }
    Point& operator=(const Point& p) { copy(p);return *this;}

protected:
    int _ix, _iy;
    real _x, _y;
};


/* A point that keep track of it's coordinates
   as double and integers
*/
class Point2DRectilinear : public Point {
public:
    Point2DRectilinear():_insidex(true),_insidey(true) {}
    bool inside() const { return _insidex&&_insidey; }
    void testx(int _m, int _M) {
	if (_ix<_m || _ix>=_M) {
	    _insidex=false;
	} else {
	    _insidex=true;
	}
    }
    void testy(int _m, int _M) {
	if (_iy<_m || _iy>=_M) {
	    _insidey=false;
	} else {
	    _insidey=true;
	}
    }
    void copy(const Point2DRectilinear& p) {
	Point::copy(p);
	_insidex = p._insidex;_insidey=p._insidey;
    }
    Point2DRectilinear& operator=(const Point2DRectilinear& p) { copy(p);return *this; }
protected:
    bool _insidex, _insidey;

};

/* A special transformation operation that transforms
   i,j (int) coordinates with a translation and scale
*/
class ScaleTransform {
public:
    typedef Point2DRectilinear  point;
    typedef point::real real;

    ScaleTransform(int _nx, int _ny,
		   real _x0, real _y0,
		   real _dx, real _dy):
	nx(_nx), ny(_ny), x0(_x0), y0(_y0), dx(_dx), dy(_dy) {}

    void set(point& p, int x, int y) const {
	p.set(x0 + x*dx, y0 + y*dy);
	p.testx(0,nx);
	p.testy(0,ny);
    }
    void incx(point& p, real k=1) const {
	p.setx( p.x() + k*dx );
	p.testx(0,nx);
    }
    void incy(point& p, real k=1) const {
	p.sety( p.y() + k*dy );
	p.testy(0,ny);
    }
public:
    int nx, ny;
    real x0, y0;
    real dx, dy;
};


class Point2D : public Point {
public:
    Point2D():_inside(true) {}
    bool inside() const { return _inside; }
    void copy(const Point2D& p) {
	Point::copy(p);
	_inside = p._inside;
    }
    Point2D& operator=(const Point2D& p) { copy(p);return *this; }

    void test(int _xm, int _xM, int _ym, int _yM) {
	if ( (_ix<_xm) || (_ix>=_xM) || (_iy<_ym) || (_iy>=_yM)) {
	    _inside=false;
	} else {
	    _inside=true;
	}
    }
    bool _inside;
};

class LinearTransform {
public:
    typedef Point2D  point;
    typedef point::real real;

    LinearTransform(int _nx, int _ny,
		    real _x0, real _y0,
		    real _xx, real _xy,
		    real _yx, real _yy):nx(_nx), ny(_ny),
					    x0(_x0), y0(_y0),
					    xx(_xx), xy(_xy),
					    yx(_yx), yy(_yy) {
    }
    void set(point& p, int x, int y) const {
	p.set(x0 + x*xx + y*xy,
	      y0 + x*yx + y*yy);
	p.test(0, nx, 0, ny);
    }
    void incx(point& p, real k=1) const {
	p.set( p.x()+k*xx, p.y()+k*yx );
	p.test(0, nx, 0, ny);
    }
    void incy(point& p, real k=1) const {
	p.set( p.x()+k*xy, p.y()+k*yy );
	p.test(0, nx, 0, ny);
    }
public:
    int nx, ny;
    real x0, y0;
    real xx, xy, yx, yy;
};

template<class axis_type>
class Point2DAxis : public Point {
public:

    Point2DAxis():_insidex(true),_insidey(true) {}
	bool inside() const { return _insidex&&_insidey; }

    void set(const axis_type& ax, real x,
	     const axis_type& ay, real y) {
	setx(ax, x);
	sety(ay, y);
    }
    void setx(const axis_type& ax, real x) {
	_ix = -1;
	_x = x;
	while(_ix<ax.ni-1 && ax.value(_ix+1)<x) {
	    ++_ix;
	}
    }
    void sety(const axis_type& ay, real y) {
	_iy = -1;
	_y = y;
	while(_iy<ay.ni-1 && ay.value(_iy+1)<y) {
	    ++_iy;
	}
    }
    void incx(const axis_type& ax, real dx) {
	_x +=dx;
	if (dx<0) {
	    while(_ix>=0 && ax.value(_ix)>=_x) {
		--_ix;
	    }
	} else {
	    while(_ix<ax.ni-1 && ax.value(_ix+1)<_x) {
		++_ix;
	    }
	}
    }
    void incy(const axis_type& ay, real dy) {
	_y += dy;
	if (dy<0) {
	    while(_iy>=0 && ay.value(_iy)>=_y) {
		--_iy;
	    }
	} else {
	    while(_iy<ay.ni-1 && ay.value(_iy+1)<_y) {
		++_iy;
	    }
	}
    }
    void copy(const Point2DAxis<axis_type>& p) {
	Point::copy(p);
	_insidex = p._insidex; _insidey=p._insidey;
    }
    Point2DAxis<axis_type>& operator=(const Point2DAxis<axis_type>& p) { copy(p);return *this; }

    bool _insidex, _insidey;
};

template<class axis_type>
class XYTransform {
public:
    static const int toto=1;
    typedef Point2DAxis<axis_type>  point;
    typedef typename point::real real;

    XYTransform(int _nx, int _ny,
		real _x0, real _y0,
		real _dx, real _dy,
		const axis_type& _ax,
		const axis_type& _ay):nx(_nx), ny(_ny), x0(_x0), y0(_y0),
	  dx(_dx), dy(_dy), ax(_ax), ay(_ay) {}

    void testx(point& p) const {
	if (p.ix()<0 || p.ix()>=nx) {
	    p._insidex=false;
	} else {
	    p._insidex=true;
	}
    }
    void testy(point& p) const {
	if (p.iy()<0 || p.iy()>=ny) {
	    p._insidey=false;
	} else {
	    p._insidey=true;
	}
    }
    void set(point& p, int x, int y) const {
	p.set(ax, x0 + x*dx,
	      ay, y0 + y*dy);
	testx(p);
	testy(p);
    }
    void incx(point& p, real k=1) const {
	p.incx( ax, k*dx );
	testx(p);
    }
    void incy(point& p, real k=1) const {
	p.incy( ay, k*dy );
	testy(p);
    }
public:
    int nx, ny;
    real x0, y0;
    real dx, dy;
    const axis_type& ax;
    const axis_type& ay;
};



#endif
