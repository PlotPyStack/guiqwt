c  Copyright © 2009-2011 CEA
c  Licensed under the terms of the CECILL License
c  (see guiqwt/__init__.py for details)

      subroutine hist2d(X, Y, n, i0, i1, j0, j1, R,
     +                  nx, ny, do_log, nmax)
c Compute a 2-D Histogram from data X(i),Y(i)
cf2py intent(in) X, Y, do_log
cf2py intent(in,out) R
cf2py intent(out) nmax
      double precision :: X(n), Y(n), R(nx,ny)
      double precision :: i0, i1, j0, j1
      integer :: n, nx, ny
      integer :: i, ix, iy
      integer :: do_log
      double precision :: nmax
      double precision :: cx, cy

      cx = nx/(i1-i0)
      cy = ny/(j1-j0)
      do i=1,n
        ix = 1+ NInt( (X(i)-i0)*cx )
        iy = 1+ NInt( (Y(i)-j0)*cy )
        if ((ix .GE. 1) .AND. (ix .LE. nx) .AND. (iy .GE. 1)
     +      .AND. (iy .LE. ny)) then
            R(ix,iy)=R(ix,iy)+1
        end if
      end do

c Apply log if needed and compute max value at the same time
      nmax = 0.0
      if (do_log.GE.1) then
         do j=1,ny
            do i=1,nx
               R(i,j)=log(1+R(i,j))
               nmax = max(nmax, R(i,j))
            end do
         end do
      else
         do j=1,ny
            do i=1,nx
               nmax = max(nmax, R(i,j))
            end do
         end do
      endif
      end subroutine

      subroutine hist2d_func(X, Y, Z, n, i0, i1, j0, j1, R, V,
     +     nx, ny, do_func)
c Compute a 2-D Histogram from data X(i),Y(i)
c do_func is a parameter selecting the computation to do
c if not specified, R receives the number of points in the bin
c    0 : V1 contains the maximum of the Z values in the bin,
c    1 : V1 contains the minimum of the Z values in the bin,
c    2 : V1 contains the sum of the Z values in the bin,
c    3 : V1 contains the product of the Z values in the bin, (V should be initialized to 1)
c    4 : V1 contains the average of the Z values in the bin,
c    5 : R contains the index of the minimum of the Z values in the bin, V1 the corresponding min
c    6 : R contains the index of the maximum of the Z values in the bin, V1 the corresponding max
c TODO : provide V1 V2 to compute min/max mean/std together

cf2py intent(in) X, Y, Z, do_func
cf2py intent(in,out) R,V
      double precision :: X(n), Y(n), Z(n), R(nx,ny), V(nx,ny)
      double precision :: i0, i1, j0, j1
      integer :: n, nx, ny
      integer :: i, ix, iy
      integer :: do_func
      double precision :: nmax
      double precision :: cx, cy

      cx = nx/(i1-i0)
      cy = ny/(j1-j0)
      do i=1,n
        ix = 1+ NInt( (X(i)-i0)*cx )
        iy = 1+ NInt( (Y(i)-j0)*cy )
        if ((ix .GE. 1) .AND. (ix .LE. nx) .AND. (iy .GE. 1)
     +      .AND. (iy .LE. ny)) then
            select case (do_func)
            case (0) ! max
               R(ix,iy)=R(ix,iy)+1
               V(ix,iy) = max(V(ix,iy), Z(i))
            case (1) ! min
               R(ix,iy)=R(ix,iy)+1
               V(ix,iy) = min(V(ix,iy), Z(i))
            case (2) ! sum
               R(ix,iy)=R(ix,iy)+1
               V(ix,iy) = V(ix,iy) + Z(i)
            case (3) ! prod
               R(ix,iy)=R(ix,iy)+1
               V(ix,iy) = V(ix,iy) * Z(i)
            case (4) ! avg
               R(ix,iy)=R(ix,iy)+1
               V(ix,iy) = V(ix,iy) + (Z(i)-V(ix,iy))/R(ix,iy)
            case (5) ! argmin
               if (V(ix,iy) .GT. Z(i)) then
                  R(ix,iy) = i
                  V(ix,iy) = Z(i)
               end if
            case (6) ! argmax
               if (V(ix,iy) .LT. Z(i)) then
                  R(ix,iy) = i
                  V(ix,iy) = Z(i)
               end if
            end select
        end if
      end do
      end subroutine
