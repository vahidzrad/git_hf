from pylab import *

sig = 1.0
E = 1.0
nu = 0.3
mu = E/(2*(1+nu))
kappa = (3-nu)/(1+nu)

# Elliptic hole geometry
u_rho = zeros(10)
rho = linspace(0.001, 0.999, 10)
for i, rho_ in enumerate(rho):
    a = 1
    b = rho_*a
    c = sqrt(a**2-b**2)
    xi0 = arccosh(a/c)

    # Spatial point
    x = 1
    y = 1

    # Conversion
    zeta = arccosh((x+y*1j)/c)
    xi = zeta.real
    eta = zeta.imag

    # Scale factor
    h = c*sqrt(sinh(xi)**2+sin(eta)**2)
    xihat1 = c*sinh(xi)*cos(eta)/h
    xihat2 = c*cosh(xi)*sin(eta)/h
    etahat1 = -c*cosh(xi)*sin(eta)/h
    etahat2 = c*sinh(xi)*cos(eta)/h

    # Displacement given by Gao, Xin-Lin. "A general solution of an infinite elastic plate with an elliptic hole under biaxial loading."
    uxi = sqrt(2)*sig/(16*mu)*sqrt(a**2-b**2)/sqrt(cosh(2*xi)-cos(2*eta))*((kappa-1)*cosh(2*xi)-(kappa+1)*cos(2*eta)+2*cosh(2*xi0)) + \
          sqrt(2)*sig/(16*mu)*sqrt(a**2-b**2)/sqrt(cosh(2*xi)-cos(2*eta))*exp(2*xi0)*((kappa-1)*exp(-2*xi)-(kappa+1)*cos(2*eta)+2*exp(-2*xi0)-2*sinh(2*(xi-xi0))*cos(2*eta))
    ueta = sqrt(2)*sig/(16*mu)*sqrt(a**2-b**2)/sqrt(cosh(2*xi)-cos(2*eta))*exp(2*xi0)*(2*cosh(2*(xi-xi0))*sin(2*eta)+2*(kappa-1)*cos(eta)*sin(eta))

    u1 = uxi*xihat1 + ueta*etahat1
    u2 = uxi*xihat2 + ueta*etahat2

    u_rho[i] = u1

plot(rho, u_rho, label="Elliptic")

# To be compared with circular ones
r = sqrt(x**2+y**2)
theta = arctan2(y, x)

ux = -sig/(8*mu)*(-r*(kappa-3)*cos(theta)+2/r*(-(1-kappa)*cos(theta)+cos(3*theta))-2/pow(r, 3)*cos(3*theta))
uy = sig/(8*mu)*(r*(kappa+1)*sin(theta)+2/r*((1+kappa)*sin(theta)-sin(3*theta))+2/pow(r, 3)*sin(3*theta))
plot([1], [ux], "o", label="Circular")
xlabel(r"$\rho$")
ylabel("$u_1$ at $(1,1)$")
show()

print("at rho = %g, x = (%g, %g)" %(rho_, x, y))
print("(u_ellip - u_circ)/u_circ = %.3e and (v_ellip - v_circ)/v_circ = %.3e" %((u1 - ux)/ux, (u2 - uy)/uy))
