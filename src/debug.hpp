/* -*- coding: utf-8;mode:c++;c-file-style:"stroustrup" -*- */
/*
  Copyright © 2009-2010 CEA
  Licensed under the terms of the CECILL License
  (see guiqwt/__init__.py for details)
*/
#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#define DEBUG 0

#if DEBUG

#define check(msg,x,n,r) if (x<0||x>=n) { printf(msg "%d out of bound (%d)\n", x, n); return r;}
#define check_img_ptr(msg,p,r,img) if (p<img.base||p>(img.base+(img.ni-1)*img.si+(img.nj-1)*img.sj)) \
    { printf(msg "%p out of bound (%p,%dx%d, %dx%d\n", p,img.base,img.ni,img.si,img.nj,img.sj); return r;}

#else

#define check(msg,x,n,r) ;
#define check_img_ptr(msg,p,r,img) ;

#endif


#endif
