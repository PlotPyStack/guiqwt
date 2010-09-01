!~ Copyright © 2009-2010 CEA
!~ Licensed under the terms of the CECILL License
!~ (see guiqwt/__init__.py for details)

module Mandel


contains

subroutine mandelbrot( orig, dx, dy, R, NX, NY, NMAX )
  complex,intent(in) :: orig, dx, dy
  integer,intent(in) :: NX, NY, NMAX
  integer*2,intent(inout) :: R(NX,NY)
  complex(kind=kind(0d0)) :: point, z, zn
  integer :: i,j,k

  point = orig
  do i=1,NX
     z = point
     do j=1,NY
        zn = z
        do k=1,NMAX
           if (real(zn)*real(zn)+imag(zn)*imag(zn) >= 5.) exit
           zn = zn*zn + z
        end do
        R(i,j) = k-1
        z = z + dy
     end do
     point = point + dx
  end do

end subroutine mandelbrot


end module Mandel
