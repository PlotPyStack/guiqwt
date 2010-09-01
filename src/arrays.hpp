/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2009-2010 CEA
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)
*/
#ifndef __ARRAYS_HPP__
#define __ARRAYS_HPP__

#include "debug.hpp"


template<class Image>
class PixelIterator
{
public:
    typedef typename Image::value_type value_type;

    PixelIterator(Image& _img):
	img(_img), cur(_img.base) {
    }

    value_type& operator()() {
	check_img_ptr("pixeliter:",cur,out,img);
	return *cur;
    }
    value_type& operator()(int dx, int dy) {
	return *(cur+dy*img.si + dx*img.sj);
    }
    void move(int dx, int dy) {
	cur += dy*img.si + dx*img.sj;
    }
    void moveto(int x, int y) {
	cur = img.base + y*img.si + x*img.sj;
    }

protected:
    Image& img;
    value_type* cur;
    value_type out;
};

template<class T>
class Array1D
{
public:
    typedef T value_type; // The type of pixel data from the image
    class iterator : public std::iterator<std::random_access_iterator_tag,T> {
    public:
	iterator():pos(0L),stride(0) {}
	iterator(const Array1D& arr):pos(arr.base),stride(arr.si) {}
	iterator(const iterator& it, int n=0):pos(it.pos),stride(it.stride) {*this+=n;}
	T& operator*() { return *pos; }
	const T& operator*() const { return *pos; }
	T& operator[](int n) { return *(pos+n*stride); }
	const T& operator[](int n) const { return *(pos+n*stride); }
	iterator& operator+=(int n) { pos+=stride*n;return *this; }
	int operator-(const iterator& it) { return (pos-it.pos)/stride; }
	iterator operator+(int n) { return iterator(*this,n); }
	iterator operator-(int n) { return iterator(*this,-n); }
	iterator& operator=(const iterator& it) { pos=it.pos;stride=it.stride;return *this; }
	iterator& operator++() { pos+=stride; return *this; }
	iterator& operator--() { pos+=stride; return *this; }
	iterator operator++(int) { iterator it(*this);pos+=stride; return it; }
	iterator operator--(int) { iterator it(*this);pos+=stride; return it; }
	bool operator<(const iterator& it) { return pos<it.pos;}
	bool operator==(const iterator& it) { return pos==it.pos;}
	bool operator!=(const iterator& it) { return pos!=it.pos;}

    protected:
	T* pos;
	int stride;
    };
    Array1D() {}
    Array1D(PyArrayObject* arr) {
	base = (value_type*)PyArray_DATA(arr);
	ni = PyArray_DIM(arr, 0);
	si = PyArray_STRIDE(arr,0)/sizeof(value_type);
    }

    Array1D( value_type* _base, int _ni, int _si):
	base(_base), ni(_ni),
	si(_si/sizeof(value_type)) {
    }
    iterator begin() { return iterator(*this); }
    iterator end() { iterator it(*this); it+=ni; return it; }
    void init( value_type* _base, int _ni, int _si) {
	base = _base; ni = _ni; si = _si;
    }
    
    // Pixel accessors
    value_type& value(int x) {
	check("array1d:",x,ni,outside);
	return *(base+x*si);
    }
    const value_type& value(int x) const {
	check("array1d:",x,ni,outside);
	return *(base+x*si);
    }

public:
    value_type outside;
    value_type* base;
    int ni; // dimensions
    int si; // strides in sizeof(value_type)
};

template<class T>
class Array2D
{
public:
    typedef T value_type; // The type of pixel data from the image

    Array2D() {}
    Array2D(PyArrayObject* arr) {
	base = (value_type*)PyArray_DATA(arr);
	ni = PyArray_DIM(arr, 0);
	nj = PyArray_DIM(arr, 1);
	si = PyArray_STRIDE(arr, 0)/sizeof(value_type);
	sj = PyArray_STRIDE(arr, 1)/sizeof(value_type);
    }

    Array2D( value_type* _base, int _ni, int _nj, int _si, int _sj):
	base(_base), ni(_ni), nj(_nj),
	si(_si/sizeof(value_type)), sj(_sj/sizeof(value_type)) {
    }
    void init( value_type* _base, int _ni, int _nj, int _si, int _sj) {
	base = _base; ni = _ni; nj = _nj; si = _si; sj = _sj;
    }
    
    // Pixel accessors
    value_type& value(int x, int y) {
	check("array2d x:",x,nj,outside);
	check("array2d y:",y,ni,outside);
	return *(base+x*sj+y*si);
    }
    const value_type& value(int x, int y) const {
	check("array2d x:",x,nj,outside);
	check("array2d y:",y,ni,outside);
	return *(base+x*sj+y*si);
    }

public:
    value_type outside;
    value_type* base;
    int ni, nj; // dimensions
    int si, sj; // strides in sizeof(value_type)
};

template<class Image>
void set_array(Image& img, PyArrayObject* arr)
{
    img.init( (typename Image::value_type*)PyArray_DATA(arr),
	      PyArray_DIM(arr, 0), PyArray_DIM(arr,1),
	      PyArray_STRIDE(arr,0)/sizeof(typename Image::value_type),
	      PyArray_STRIDE(arr,1)/sizeof(typename Image::value_type) );
}

#endif
