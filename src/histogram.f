c  Copyright © 2009-2010 CEA
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
         do i=1,nx
            do j=1,ny
               R(i,j)=log(1+R(i,j))
               if (R(i,j).GT.nmax) then
                  nmax = R(i,j)
               end if
            end do
         end do
      else
         do i=1,nx
            do j=1,ny
               if (R(i,j).GT.nmax) then
                  nmax = R(i,j)
               end if
            end do
         end do
      endif
      end subroutine
