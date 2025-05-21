import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as sp

energies = np.array(range(48,60+1)) #MeV
yy = []

# loading data (energies)
dir = "dati_fluka_old"
fileList =[f for f in os.listdir(dir) if f.endswith(".dat")]

i = 0
for file in fileList:
    path = os.path.join(dir, file)

    col = np.loadtxt(path,skiprows=1,usecols=2)
    # loading data (xx)
    if i == 0:
        xx_1 = np.loadtxt(path,skiprows=1,usecols=0)
        xx_2 = np.loadtxt(path,skiprows=1,usecols=1)
        xx = (xx_1 + xx_2)/2
        rows = len(xx)
        print(rows)

        #initialize array to fetch data
        yy = np.zeros([len(energies), rows])
    yy[i, :] = col.transpose()
    i += 1

# GeV --> MeV
yy = yy*10e3


### calcs

#def
weights_0 = np.ones(len(energies))
ref = np.max(yy[-1,:])

yy_extr = yy[:,40*50:60*50]

# plot
yy_sum = np.sum(yy, axis=0)
for i in range(len(energies)):
    plt.plot(xx,yy[i,:],lw='0.5')
#plt.plot(xx, yy_sum)
plt.xlabel("[cm]")
plt.ylabel("[MeV]")
plt.show()

def resto(w):
    return (ref - np.sum(yy_extr * w[:, np.newaxis], axis=0))

res = sp.optimize.least_squares( resto,weights_0 )
wf = res.x
print(wf)

#wf[-1] = wf[-1]*0.1
fun_final = yy*wf[:,np.newaxis]
plt.plot(xx,np.sum(fun_final,axis=0),lw='0.5')
plt.show()