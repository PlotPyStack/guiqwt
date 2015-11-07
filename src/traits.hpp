/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2009-2010 CEA
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)

  
  Two lines of the following code are distributed under LGPL license terms and 
  with a different copyright. These two lines are the Visual Studio x86_64 
  (_M_X64) inline versions of lrint() and lrintf() functions, and were adapted 
  from fast_convert.h (SpanDSP), which is:

  Copyright © 2009 Steve Underwood
  All rights reserved.
 
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License version 2.1,
  as published by the Free Software Foundation.
 
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.
  
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

/* MSVC does not have lrint/lrintf */
#ifdef _MSC_VER
    #include <intrin.h>
    __inline long int lrint (double f)
    {
    #ifdef _M_X64
        return (long int)_mm_cvtsd_si64x(_mm_loadu_pd((const double*)&f));
    #else
        int i; 
        _asm
        {
            fld f
            fistp i
        }
        return i;
    #endif
    }

    __inline long int lrintf (float f)
    {
    #ifdef _M_X64
        return _mm_cvt_ss2si(_mm_load_ss((const float*)&f));
    #else
        int i;
        _asm
        {
            fld f
            fistp i
        }
        return i;
    #endif
    }
#endif

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
    case NPY_UINT64:
	algo.template run<npy_uint64>();
	break;
    case NPY_INT64:
	algo.template run<npy_int64>();
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
