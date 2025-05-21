import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize


# Definizione delle funzioni gaussiane (come picchi)
def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def generate_functions(x, centers, sigma):
    return np.array([gaussian(x, mu, sigma) for mu in centers])

xx = np.linspace(1, 5, 100)
centers = [3, 3.5, 4, 4.5]
sigma = 0.5
funcs = generate_functions(xx, centers, sigma)

funcs[1,:] *= 1.5
funcs[3,:] *= 1.3




fun_sum = np.sum(funcs,axis=0) # !!!
for i in range(4):
    plt.plot(xx,funcs[i,:])
plt.plot(xx,fun_sum)
plt.show()

weights_0 = np.ones(len(centers))
ref = 1

funcs_ext = funcs[:,50:95]


def resto(w):
    return ( ref - np.sum( funcs_ext*w[:,np.newaxis],axis=0 ) ) ## IMPORTANTE AXIS ZEROOOOOOOOOOO

res = scipy.optimize.least_squares( resto,weights_0 )
wf = res.x
print(wf)

fun_final = funcs*wf[:,np.newaxis]
plt.plot(xx,np.sum(fun_final,axis=0))
plt.show()