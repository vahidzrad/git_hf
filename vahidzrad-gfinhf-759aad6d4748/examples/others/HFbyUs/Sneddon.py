import numpy as np
import matplotlib.pyplot as plt
from dolfin import *



def SneddonWidth(p_max):
	#p_max=0.250
	sigma_min=0.
	nu=0.0
	E=1.
	l_crack=0.2

	x=np.linspace(0.0, l_crack, 100)
	x_=np.linspace(-l_crack, 0.0, 100)

	delta_P=p_max-sigma_min
	E_prime=E/(1-nu**2)
	b=2*delta_P*l_crack/E_prime

	uy=b*(1-x**2/l_crack**2)**0.5
	uy_=b*(1-x_**2/l_crack**2)**0.5


	x=x+2
	x_=x_+2
	return x, x_, uy, uy_

if __name__ == '__main__':
        SneddonWidth()


#plt.plot(X, Width,'r-')
#save_fig = "results/"
#plt.savefig(save_fig + "CrkOpnSneddon.pdf")


