	program ulvz_layer_1d
	parameter(n=256)
	real h(-2:n+1),v(-2:n+1),d(-2:n+1),p(-2:n+1),s(-2:n+1),f(-2:n+1)
	real a(-2:n+1),b(-2:n+1),c(-2:n+1),r(-2:n+1),w(-2:n+1),q(-2:n+1) ! Work arrays

	Du=1.
	Dp=0.01
	
	dx=(1.d0/n)
	nc=0 ! Flag for Cartesian (0) vs. axi-symmetric (1)

	nsteps=100000000
	time=0.d0
	timelast=0.
	dtwrite=0.005d0
	twrite=dtwrite
	timemax=2.0d0
	print*,'Total number of write steps:',timemax/dtwrite

	call initial_h(h,n,dx)
	call write_step(h,0,n,dx,'h')

	do i=1,nsteps
	  iwriteflag=0
	  call input_v(v,n,dx,nc)
	  call input_d(d,h,w,n,dx,Du,nc)
	  call input_p(p,n,dx,Dp,nc)
	  call decide_dt(v,d,n,dx,dt,nc)
	  timelast=time
	  time=time+dt
        
	  if(time.gt.twrite) then
	    dt=twrite-timelast
	    time=twrite
	    print*, time
	    iwriteflag=1
	    twrite=twrite+dtwrite
	  endif
	  call superbee(h,v,s,f,w,n,dx,dt,nc)
	  call impl_diff(h,d,p,a,b,c,r,w,n,dx,dt,nc)
	  call add_pressure_rhs(h,d,p,n,dx,dt,nc)
c	  if(mod(i,iwrite).eq.0) then
	  if(iwriteflag.eq.1) then
	    iwrite=int(time/dtwrite)
	    print*,'Step:',iwrite
	    call volume_int(h,n,volume,dx,nc)
	    print*,'Volume:',volume
	    call val_max(h,n,hmax)
	    print*,'max height:',hmax
	    call val_max(d,n,dmax)
	    print*,'max diff:',dmax
	    call write_step(h,iwrite,n,dx,'h')
c	    call input_test(q,n,dx,time) ! This was just a test function
c	    call write_step(q,iwrite,n,dx,'q') ! which was written to compare
	  endif
	  if(time.gt.timemax) stop
	enddo

	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C INITIALIZATION ROUTINES
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine initial_h(h,n,dx)
	real h(-2:n+1)

	pi=4.0*atan(1.0)
	do i=-1,n+1
	  x=(0.5d0+(1.0*i))*dx
	  h(i)=1.!+cos(8.*pi*x)
c	  if(i.gt.n/2) h=1.
	enddo

	return
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine decide_dt(v,d,n,dx,dt,nc)
	real v(-2:n+1),d(-2:n+1)

c This routine decides the time step to be taken...
c Need to adapt to the evolving solution and changing stability conditions

	if(nc.eq.0) then
	  call val_max(v,n,vmax)
	  call val_max(d,n,dmax)
	else
	  call val_max_axi(v,n,vmax,dx)
	  call val_max_axi(d,n,dmax,dx)
	endif

	dt_courant=dx/vmax
	dt_expl_diff=0.5*dx*dx/dmax

	dt=min(dt_courant,dt_expl_diff)
	
	dt=0.5*dt ! No reason to push things to the limit, take a chunk off for extra stability/accuracy

	return
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc 


	subroutine input_test(q,n,dx,time)
	real q(-2:n+1)

	pi=4.0*atan(1.0)
	factor=-64.*pi*pi*time
	do i=-1,n+1
	  x=(0.5d0+(1.0*i))*dx
	  q(i)=1.1+exp(factor)*cos(8.*pi*x)
	enddo

	return
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine input_v(v,n,dx,nc)
	real v(-2:n+1)

	r_u=0.1
	pi=4.0*atan(1.0)

	if(nc.eq.0) then
	  do i=-1,n+1
	    x=(1.0*i)*dx
	    v(i)=-sin(pi*x)
	  enddo
	else
	  do i=-1,n+1
	    x=(1.0*i)*dx
c	    v(i)=-sin(pi*x)
	    
c Functions similar to -sin(pi*x)
		v(i)=-4.*x*(1-x)
c		v(i)=-2.6*(x-x**3)
	    	    
c	    v(i)=-sqrt(abs(sin(pi*x)))! r_u/(r_u+x)

c Hier-Majumder (right and wrong)	    
c	    v(i)=-r_u/(r_u+x)
c	    v(i)=-r_u/(r_u+x)+log(r_u+x)-log(r_u)+1.d0
	    v(i)=x*v(i)
	  enddo
	endif

	return
	end
	
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine input_d(d,h,w,n,dx,Du,nc)
	real d(-2:n+1),h(-2:n+1),w(-2:n+1)

	call boundaries(h,n)

c First input the values at the cell centers into the work array...
	do i=-1,n+1
	  w(i)=Du*h(i)*h(i)*h(i)
	enddo

	tolerance=1.
c Now compute the values at the cell faces...
	if(nc.eq.0) then
	  do i=0,n
	    diff=abs(log(w(i+1)/w(i)))
	    if(diff.lt.tolerance) then
	      d(i)=2.*w(i+1)*w(i)/(w(i+1)+w(i))
	    else
	      d(i)=0.5*(w(i+1)+w(i))
	    endif
	  enddo
	else
	  do i=0,n
	    x=(1.0*i)*dx
	    diff=abs(log(w(i+1)/w(i)))
	    if(diff.lt.tolerance) then
	      d(i)=2.*x*w(i+1)*w(i)/(w(i+1)+w(i))
	    else
	      d(i)=0.5*x*(w(i+1)+w(i))
	    endif
	  enddo
	endif

c	do i=-1,n+1
c	  d(i)=0.1
c	enddo

c Impose insulating side walls
	d(0)=0.d0
	d(n)=0.d0

	return
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine input_p(p,n,dx,Dp,nc)
	real p(-2:n+1)

	pi=4.0*atan(1.0)
	r_u=0.1
	if(nc.eq.0) then
	  do i=-1,n+1
	    x=(1.0*i)*dx
	    p(i)=-pi*cos(pi*x)*Dp
	  enddo
	else
	  do i=-1,n+1
	    x=(1.0*i)*dx
c	    p(i)=-pi*cos(pi*x)*Dp
	    
c Functions similar to -sin(pi*x)
		p(i)=(-4.+8.*x)*Dp
c		p(i)=(-2.6+7.8*x**2)*Dp	    

c Hier-Majumder	    
c	    p(i)=Dp*(2.*r_u+x)/(r_u+x)**2
	  enddo	
	endif

	p(-1)=p(0)
	p(n)=p(n-1)

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C ADVECTION ROUTINES
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine superbee(h,v,s,f,w,n,dx,dt,nc)
	real h(-2:n+1),v(-2:n+1),s(-2:n+1),f(-2:n+1),w(-2:n+1)

	call boundaries(h,n)
c	call boundary_wrap(h,n)
c	call val_max(v,n,vmax)

c	call get_slopes(h,s,n)
	call get_flux(f,v,h,n,dx,dt)
	if(nc.eq.0) then
	  do i=0,n-1
	    h(i)=h(i)-dt*(f(i+1)-f(i))/dx
	  enddo
	else
	  do i=0,n-1
	    x=(0.5d0+1.0*i)*dx
	    h(i)=h(i)-dt*(f(i+1)-f(i))/(x*dx)
	  enddo
	endif

	call boundaries(h,n)

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine boundaries(h,n)
	real h(-2:n+1)
	
	h(-2)=h(0)
	h(-1)=h(0)
	h(n)=h(n-1)
	h(n+1)=h(n-1)

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine boundary_wrap(h,n)
	real h(-2:n+1)
	
	h(-2)=h(n-2)
	h(-1)=h(n-1)
	h(n)=h(0)
	h(n+1)=h(1)

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine get_flux(f,v,h,n,dx,dt)
	real f(-2:n+1),v(-2:n+1),h(-2:n+1)

	tolerance=1.e-4
	do i=0,n
	  theta=-1.
	  dh=abs((h(i)-h(i-1))/h(i))
	  if(v(i).ge.0.) theta=1.
	  if(theta.ge.0) then
	    r=(h(i-1)-h(i-2))/(h(i)-h(i-1))
	  else
	    r=(h(i+1)-h(i))/(h(i)-h(i-1))
	  endif
	  if(dh.lt.tolerance) r=1000.
	  fdc=v(i)*((1.+theta)*h(i-1)+(1.-theta)*h(i)) ! Donor Cell Part
	  f(i)=fdc+abs(v(i))*(1.-abs(v(i)*dt/dx))*phi(r)*(h(i)-h(i-1)) ! Flux Limiting Part
	  f(i)=0.5*f(i)
c	  print*,'flux',i,r,f(i),v(i)
	enddo

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	real function phi(r)

	phi=max(0.,min(1.,2.*r),min(2.,r))
c	phi=fminmod(1.,r)

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine get_slopes(h,s,n)
	real h(-2:n+1),s(-2:n+1)

	d1x=(1.0*n)
	do i=0,n
	  dhdxp=d1x*(h(i+1)-h(i))
	  dhdxm=d1x*(h(i)-h(i-1))
	  s1=fminmod(dhdxp,2.*dhdxm)
	  s2=fminmod(2.*dhdxp,dhdxm)
	  s(i)=fmaxmod(s1,s2) ! Superbee choice of slopes
	enddo

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	real function fminmod(a,b)

	if(a*b.le.0.) then
	  fminmod=0.
	elseif(abs(a).ge.abs(b)) then
	  fminmod=b	
	else
	  fminmod=a
	endif

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	real function fmaxmod(a,b)

	if(a*b.le.0.) then
	  fmaxmod=0.
	elseif(abs(a).ge.abs(b)) then
	  fmaxmod=a
	else
	  fmaxmod=b
	endif

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C DIFFUSION ROUTINES
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine impl_diff(h,d,p,a,b,c,r,w,n,dx,dt,nc)
	real h(-2:n+1),d(-2:n+1),p(-2:n+1)
	real a(-2:n+1),b(-2:n+1),c(-2:n+1),r(-2:n+1),w(-2:n+1) ! Work arrays

	call boundaries(h,n)

c Discretization factor...
	dtx=dt/(dx*dx)

c Input zero flux boundary conditions explicitly as part of the tri-diagonal matrix equations...
	a(-1)=0.d0
	b(-1)=1.d0
	c(-1)=-1.d0
	r(-1)=0.d0
	a(n)=-1.d0
	b(n)=1.d0
	c(n)=0.d0
	r(n)=0.d0

c Fill in the tri-diagonal matrix stencils for the points inside the domain
	if(nc.eq.0) then
	  do i=0,n-1
	    a(i)=-dtx*d(i)
	    b(i)=1.+dtx*(d(i)+d(i+1))
	    c(i)=-dtx*d(i+1)
	    r(i)=h(i)
c	    print*,'in-tri',i,dtx,d(i),a(i),b(i),c(i),r(i)
	  enddo
	elseif(nc.eq.1) then
	  do i=0,n-1
	    x=(0.5d0+1.0*i)*dx
	    dtxas=dtx/x
	    a(i)=-dtxas*d(i)
	    b(i)=1.+dtxas*(d(i)+d(i+1))
	    c(i)=-dtxas*d(i+1)
	    r(i)=h(i)
c	    print*,'in-tri',i,dtx,d(i),a(i),b(i),c(i),r(i)
	  enddo
	endif

	call tridiagonal_solve(a,b,c,r,h,w,n)
	call boundaries(h,n)

	return
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine tridiagonal_solve(a,b,c,r,h,w,n)
	real a(-2:n+1),b(-2:n+1),c(-2:n+1),r(-2:n+1),h(-2:n+1),w(-2:n+1)
c Simple solver for tridiagonal equations
c a(i),b(i),c(i) are the stencils for u(i-1),u(i),u(i+1) respectively
c b is assumed non-zero everywhere, and greater in magnitude than a or c
c r is the right hand side
c h is the solution vector, w is a work array.

	f=b(-1)
	h(-1)=r(-1)/f
	do ii=0,n
	  w(ii)=c(ii-1)/f
	  f=b(ii)-a(ii)*w(ii) ! Should be non-zero since Eqns are diag dominant
	  h(ii)=(r(ii)-a(ii)*h(ii-1))/f
c	  print*,'hello',ii,h(ii),a(ii),b(ii),c(ii),r(ii)
	enddo
c	h(n)=0.d0
	do ii=n-1,-1,-1
	  h(ii)=h(ii)-w(ii+1)*h(ii+1)
	enddo

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C PRESSURE RIGHT HAND SIDE
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine add_pressure_rhs(h,d,p,n,dx,dt,nc)
	real h(-2:n+1),d(-2:n+1),p(-2:n+1)

	d2x=dx*dx
	if(nc.eq.0) then
	  do i=0,n-1
	    pterm=(d(i+1)*(p(i+1)-p(i))+d(i)*(p(i-1)-p(i)))/d2x
	    h(i)=h(i)+dt*pterm
	  enddo
	else
	  do i=0,n-1
	    x=(0.5d0+1.0*i)*dx
	    pterm=(d(i+1)*(p(i+1)-p(i))+d(i)*(p(i-1)-p(i)))/d2x
	    h(i)=h(i)+dt*pterm/x
	  enddo
	endif

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C UTILITY ROUTINES
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine val_max(v,n,vmax)
	real v(-2:n+1)

	vmax=0.d0
	do i=0,n
	  val=abs(v(i))
	  if(val.gt.vmax) vmax=val
	enddo

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine val_max_axi(v,n,vmax,dx)
	real v(-2:n+1)

	vmax=0.d0
	do i=1,n
	  x=(1.*i)*dx
	  val=abs(v(i))/x
	  if(val.gt.vmax) vmax=val
	enddo

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine volume_int(h,n,volume,dx,nc)
	real h(-2:n+1)

	volume=0.d0
	if(nc.eq.0) then
	  do i=0,n-1
	    volume=volume+h(i)*dx
	  enddo
	else
	  do i=0,n-1
	    x=(1.*i+0.5)*dx
	    volume=volume+x*h(i)*dx
	  enddo
	endif

	return
	end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C INPUT/OUTPUT ROUTINES
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


	subroutine write_step(h,istep,n,dx,flabel)
	real h(-2:n+1)
	character*1 flabel
	character*5 g
	character*10 fname

	call int_to_char(istep,g)
	fname(1:1)=flabel(1:1)
	fname(2:6)=g(1:5)
	fname(7:10)='.jwh'
	open(1,file=fname)
	do i=-1,n
	  x=(1.0*i+0.5)*dx
	  write(1,2) x,h(i)
2	  format(1x,f16.7,1x,f16.7)
	enddo
	close(1)

	return
	end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
	
	subroutine int_to_char(n,g)
c  Routine for converting the integer "n" to a character
c  string "g" of length "ilength," which is useful for 
c  writing file numbers in time dependent computations, etc..
c  You can change the length of the string simply by changing
c  the value of ilength to the desired length as well as 
c  changing the length of the character string in the declaration
c  to the same size.
	integer n,lev,i,ich,npart,int,ilength
	real*8 x
c  Change the "5" in the following two lines to whatever length to
c  use longer character strings (i.e. higher numbers).
	character*5 g
	ilength=5

	x=log10(1.0*n)
	if(x.ge.ilength) then
	  print*,'Integer size too large in int_to_char:'
	  print*,'Modify character length in subroutine'
	  stop
	endif

	npart=0
	do i=ilength,1,-1
	  j=ilength-i+1
	  lev=10**(i-1)
	  int=0
	  ich=n-npart
	  if(ich.ge.0.and.ich.lt.1*lev) then
	    g(j:j)='0'
	    int=0
	  elseif(ich.ge.1*lev.and.ich.lt.2*lev) then
	    g(j:j)='1'
	    int=1*lev
	  elseif(ich.ge.2*lev.and.ich.lt.3*lev) then
	    g(j:j)='2'
	    int=2*lev
	  elseif(ich.ge.3*lev.and.ich.lt.4*lev) then
	    g(j:j)='3'
	    int=3*lev
	  elseif(ich.ge.4*lev.and.ich.lt.5*lev) then
	    g(j:j)='4'
	    int=4*lev
	  elseif(ich.ge.5*lev.and.ich.lt.6*lev) then
	    g(j:j)='5'
	    int=5*lev
	  elseif(ich.ge.6*lev.and.ich.lt.7*lev) then
	    g(j:j)='6'
	    int=6*lev
	  elseif(ich.ge.7*lev.and.ich.lt.8*lev) then
	    g(j:j)='7'
	    int=7*lev
	  elseif(ich.ge.8*lev.and.ich.lt.9*lev) then
	    g(j:j)='8'
	    int=8*lev
	  elseif(ich.ge.9*lev.and.ich.lt.10*lev) then
	    g(j:j)='9'
	    int=9*lev
	  else
	    print*,'problem in int_to_char i=',i
	    stop
	  endif
	  npart=npart+int
	enddo
	
	return
	end
