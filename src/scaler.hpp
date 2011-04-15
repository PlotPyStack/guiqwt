/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2010-2011 CEA
  Ludovic Aubry
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)
*/

#ifndef _SCALER_HPP
#define _SCALER_HPP

#include "points.hpp"
#include "arrays.hpp"

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
    LinearScale(double _a, double _b, D _bg, bool apply_bg):a(_a), b(_b), bg(_bg), has_bg(apply_bg) {}

    D eval(T x) const {
	return a*x+b;
    }
    void set_bg(D& dest) const {
	if (has_bg) dest = bg;
    }
protected:
    D a, b;
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

template<class T, class D>
class NoScale
{
public:
    typedef T source_type;
    typedef D dest_type;
    NoScale(dest_type _bg, bool apply_bg):bg(_bg), has_bg(apply_bg) {}

    dest_type eval(source_type x) const { return x; }
    void set_bg(dest_type& dest) const { if (has_bg) dest = bg; }
protected:
    dest_type bg;
    bool has_bg;
};

bool check_arrays(PyArrayObject* p_src, PyArrayObject *p_dest);
bool check_lut(PyArrayObject *p_lut);


#endif // _SCALER_HPP
