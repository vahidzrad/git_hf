#!/usr/bin/env python
## @copyright M.M. Chiaramonte and Yongxing Shen and Adrian J. Lew 
## @license Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, merge, publish, 
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

## The solution to the circular arc crack. A synthesized description of the solution
#  can be found in : Chiaramonte MM, Shen Y, Keer LM, Lew AJ. "Computing stress 
#   intensity factors for curvilinear fractures"
#  @param[in] theta the half angle of the crack - optional default \f$\pi/2\f$
#  @param[in] radius the radius of the circular arc - optional default 1
#  @param[in] pxx the far field xx component of the stress tensor - optional default 1
#  @param[in] pyy the far field yy component of the stress tensor - optional defaul 1
#  @param[in] pxy the far field xy component of the stress tensor - optional defaul 0
#  @param[in] mu shear modulus - optional default 1.
#  @param[in] kappa constant as defined in paper section 2 paragraph 2- optional defaul 3.
#  @return the displacement field and the stress tensor field in cartesian basis over the complex plane, 
#  as well as the SIF mode I, SIF mode II
def get_circular_arc_crack_solution( theta = np.pi/2. , radius=1. , pxx = 1. , pyy = 1. , pxy = 0 , mu=1.,kappa =3. ): 
	#! The codinates of the crack tips for the unit circle
	
	a = np.exp( -1j*theta)
	b = np.exp( +1j*theta)

	#! The constants (7) - (10)
	
	Gamma  = ( pxx + pyy )/4. 
	Gammap = ( pyy - pxx )/2. + 1j*pxy

	C0 =  1./2 *( Gammap -  Gammap.conjugate() )*pow(np.sin(theta/2),2) +\
						 ( 4.*Gamma +( Gammap + Gammap.conjugate() )*pow(np.sin(theta/2) * np.cos(theta/2),2) )\
						 /( 2.*(1. + pow(np.sin( theta/2),2) ) )
	C1 =  -C0 * np.cos( theta)  

	D0 = 2.*Gamma  - C0
	D1 = - Gammap.conjugate() *np.cos( theta )
	D2 = Gammap.conjugate() 

	#! The branch cut and the map

	G = lambda z:  -1. if ( ( z.real <= np.cos(theta)  or z.real**2. + z.imag**2. <= 1.)\
								and  np.sqrt(z-a).imag >0 and np.sqrt(z-b).imag <0    )	else 1. 

	X = lambda z: np.sqrt( z - a )*np.sqrt( z - b ) *G(z) + 1e-6

	DX = lambda z: (z - np.cos(theta) ) /X(z)
						 	
	#! The complex potententials (9),(10)
	PhiOrigin = Gamma - C0*pow(np.sin( theta/2. ),2) + Gammap.conjugate() * pow( np.sin(theta),2) /4.

	Phi = lambda z: ( (C0*z + C1 + D1 /z + D2 / pow( z , 2.) )/ ( 2. * X(z) ) \
						 + D0/2. + Gammap.conjugate()/( 2.*pow( z, 2. ) ) )  if  abs(z)**2 > 1.e-9 else PhiOrigin

	Omega = lambda z: (C0 *z+ C1 + D1 /z + D2 / pow(z,2) )/ ( 2. * X(z) )  \
							- D0/2  - Gammap.conjugate()/( 2.*pow( z, 2. ) )

	#! The derivative of the potential (12)

	DPhi = lambda z:	-( z- np.cos( theta) )*( C0 *z + C1 + D1/z + D2/pow(z,2) )/(2. * pow( X(z),3 ) )\
							+(C0 - D1/pow(z, 2.) - 2.* D2 /pow( z, 3 ) ) / ( 2. * X(z) )\
							- Gammap.conjugate()/pow(z,3)

	#! The other stress function (6),(13c)
	PsiOrigin =	3./4. *C0*np.cos(theta)*pow(np.sin(theta), 2.)\
						+ 1./4*C0.conjugate()*pow(np.sin(theta), 2.)\
						+ Gammap *pow(np.cos( theta/2. ), 2. )\
						+ 3./16* Gammap.conjugate() * pow(np.sin(theta) ,2.) *( 5.*pow(np.sin(theta),2.) - 4. )

	#! Following (6) 
	Psi = lambda z: ( 1./pow(z, 2.) *Phi(z) - 1./pow( z, 2.)* ( Omega(1./z.conjugate() ) ).conjugate()\
						 - 1./z*DPhi(z) ) if  abs(z)**2 > 1.e-9 else PsiOrigin

	#! Following (6')
	Psi = lambda z: ( - ( DPhi(z) + (C0.conjugate()/2.)*DX(z) )/z + ( Phi(z) + Gamma + \
						 (C0.conjugate()/2.)*( X(z)-1. ) )/pow(z,2.) + Gammap/2.*(1. + DX(z) )  ) if  abs(z)**2 > 1.e-9 else PsiOrigin

	#! The integrated potentials (14) 
	vphi = lambda z: ( Gamma*z +  C0/2.*( 1 - z + X(z) )\
						 - Gammap.conjugate()/2.*( (1.+X(z) )/z - np.cos(theta) )  ) if   abs(z)**2 > 1.e-9 else 0 

	vpsi = lambda z:(- ( Phi(z) + Gamma + ( C0.conjugate()/2. )*(X(z) -1 ) )/z\
						 + Gammap/2.*( 1 + z + X(z) )\
						 - C0/2.*pow(np.sin(theta),2)\
						 + Gammap.conjugate()/2.*np.cos(theta)*pow( np.sin(theta) ,2)\
						 + C0.conjugate()/2.*np.cos(theta) ) if  abs(z)**2 > 1.e-9 else 0

	#! The displacements (1)
	alpha = lambda z: 1./(2.*mu ) * ( kappa*vphi(z) - z* Phi(z).conjugate() - vpsi( z).conjugate() )

	ux = lambda z: alpha(z).real
	uy = lambda z: alpha(z).imag

	displ = lambda z: np.array([ux(z/radius),uy(z/radius)])
	
	#! The stress  (3)

	stressxy = lambda z: ( (  z.conjugate()*DPhi(z) if  abs(z)**2 > 1.e-9 else 0 ) + Psi(z) ).imag

	stressyy = lambda z: ( 2.*Phi(z).real + ( (  z.conjugate()*DPhi(z) if abs(z)**2 > 1.e-9 else 0 )+ Psi(z) ).real )

	stressxx = lambda z: 4.*Phi(z).real - stressyy(z) 

	stress = lambda z: np.array( [ [ stressxx(z/radius) , stressxy(z/radius) ],[stressxy(z/radius) , stressyy(z/radius) ] ] )

	#! The stress intensity factors ( Cotterel and Rice 1980 )
	KI =  np.sqrt(np.pi*radius*np.sin(theta) )*(\
			( ( pyy+ pxx )/2. + (pyy-pxx)/2.*pow(np.sin(theta/2)*np.cos(theta/2),2.) )*np.cos(theta/2.)\
			/(1.+pow( np.sin(theta/2.),2.) ) \
			- (pyy -pxx)/2.*np.cos(3.*theta/2.) + pxy*( np.sin(3./2*theta) + pow(np.sin(theta/2),3) ) )
	
	KII =  np.sqrt(np.pi*radius*np.sin(theta) )*(\
			( ( pyy+ pxx )/2. + (pyy-pxx)/2.*pow(np.sin(theta/2)*np.cos(theta/2),2.) )*np.sin(theta/2.)\
			/(1.+pow( np.sin(theta/2.),2.) ) - (pyy -pxx)/2.*np.sin(3.*theta/2.) \
			 - pxy*( np.cos(3./2.*theta) + np.cos(theta/2)*pow(np.sin(theta/2.),2) ) ) 

	return displ, stress, KI, KII
