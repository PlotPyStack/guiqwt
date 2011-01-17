/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2009-2010 CEA
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)
*/
#ifndef __TRAITS_HPP__
#define __TRAITS_HPP__

#include <math.h>

/* this class (real_trait) is used
   to make the code somewhat independent of
   the real type used (float or double).
   we don't provide a template parameter
   on the Point* class to avoid clutter but
   changing a simple typedef is all it takes
   to switch from float to double type
*/

typedef int fixed;

template<typename T>
struct num_trait {
    typedef T value_type;
    typedef long large_type;
    static int toint(value_type v) { return (int)v; }
    static value_type fromdouble(double v) { return (value_type)v; }

    static const bool is_integer = true;
};

template<>
struct num_trait<float> {
    typedef float value_type;
    typedef float large_type;
    static int toint(value_type v) { return lrintf(v); }
    static value_type fromdouble(double v) { return (value_type)v; }

    static const bool is_integer = false;
};

template<>
struct num_trait<double> {
    typedef double value_type;
    typedef double large_type;
    static int toint(value_type v) { return lrint(v); }
    static value_type fromdouble(double v) { return (value_type)v; }

    static const bool is_integer = false;
};

template<>
struct num_trait<fixed> {
    typedef fixed value_type;
    typedef fixed large_type;
    static int toint(value_type v) { return (v>>15); }
    static value_type fromdouble(double v) { return lrint(v*32768.); }

    static const bool is_integer = false;
};


template<class A>
static void dispatch_array(int npy_type, A& algo) {
    switch(npy_type) {
    case NPY_FLOAT32:
	algo.template run<npy_float32>();
	break;
    case NPY_FLOAT64:
	algo.template run<npy_float64>();
	break;
    case NPY_UINT32:
	algo.template run<npy_uint32>();
	break;
    case NPY_INT32:
	algo.template run<npy_int32>();
	break;
    case NPY_UINT16:
	algo.template run<npy_uint16>();
	break;
    case NPY_INT16:
	algo.template run<npy_int16>();
	break;
    case NPY_UINT8:
	algo.template run<npy_uint8>();
	break;
    case NPY_INT8:
	algo.template run<npy_int8>();
	break;
    case NPY_BOOL:
	algo.template run<npy_bool>();
	break;
    }
}
#endif
