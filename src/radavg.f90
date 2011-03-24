!~ Copyright © 2009-2011 CEA
!~ Licensed under the terms of the CECILL License
!~ (see guiqwt/__init__.py for details)

module RadialAverage

contains

subroutine radavg(ysection, ny, yw, nyw, image, n, m, ic, jc, radius)
  integer,intent(in) :: ny, n, m, ic, jc, radius
  double precision,intent(in) :: image(0:n-1, 0:m-1)
  double precision,intent(inout) :: ysection(0:ny-1), yw(0:nyw-1)
  integer :: i,j,r

  yw = 0
  do i=max(ic-radius, 0),min(ic+radius, n-1)
     do j=max(jc-radius, 0),min(jc+radius, m-1)
        r = floor(sqrt(real((i-ic)**2+(j-jc)**2)+0.5))
        if (r <= radius) then
            ysection(r) = (yw(r)*ysection(r)+image(i,j))/(yw(r)+1)
            yw(r) = yw(r)+1
        end if
     end do
  end do

end subroutine radavg

subroutine radavg_mask(ysection, ny, yw, nyw, image, n, m, mask, nm, mm, ic, jc, radius)
  integer,intent(in) :: ny, n, m, ic, jc, radius
  double precision,intent(in) :: image(0:n-1, 0:m-1)
  integer,intent(in) :: mask(0:nm-1, 0:mm-1)
  double precision,intent(inout) :: ysection(0:ny-1), yw(0:nyw-1)
  integer :: i,j,r

  yw = 0
  do i=max(ic-radius, 0),min(ic+radius, n-1)
     do j=max(jc-radius, 0),min(jc+radius, m-1)
        r = floor(sqrt(real((i-ic)**2+(j-jc)**2)+0.5))
        if (r <= radius .and. mask(i,j) == 0) then
            ysection(r) = (yw(r)*ysection(r)+image(i,j))/(yw(r)+1)
            yw(r) = yw(r)+1
        end if
     end do
  end do

end subroutine radavg_mask

end module RadialAverage
