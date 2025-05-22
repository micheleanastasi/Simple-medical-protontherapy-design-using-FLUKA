import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as sp

energies = np.array(range(48,60+1)) #MeV
yy = []
xx = []

### LOADING DATA

dir = "dati_fluka"
fileList =[f for f in os.listdir(dir) if f.endswith(".dat")]

i = 0
for file in fileList:
    path = os.path.join(dir, file)

    col = np.loadtxt(path,skiprows=1,usecols=2)
    # loading data (xx) - fetching info about num. of rows
    if i == 0:
        xx_1 = np.loadtxt(path,skiprows=1,usecols=0)
        xx_2 = np.loadtxt(path,skiprows=1,usecols=1)
        xx = (xx_1 + xx_2)/2
        rows = len(xx)

        #initialize array to fetch data
        yy = np.zeros([len(energies), rows])
    yy[i, :] = col.transpose()
    i += 1

# GeV --> MeV
yy = yy*10e3
#print(np.argmax(yy[-1,:]))

### CALCULATIONS

#def
a = 2000
b = 3000
yy_extr = yy[:,a:b]

weights_0 = np.ones(len(energies))
ref = np.max(yy[-1,:])


# plot
yy_sum = np.sum(yy, axis=0)
for i in range(len(energies)):
    plt.plot(xx,yy[i,:],lw='0.5')
plt.vlines(a/1e3,0,np.max(yy),linestyles='dashdot',linewidth=0.5,colors="black")
plt.vlines(b/1e3,0,np.max(yy),linestyles='dashdot',linewidth=0.5,colors="black")
plt.title("Pristine Bragg Peaks + range used to get SOBP")
plt.xlabel("[cm]")
plt.ylabel("[MeV/cm]")
plt.savefig("saved_plots/pristinePeaks.png",dpi=300)
plt.show()

# define functions to minimize peaks differences
def resto(w):
    return (ref - np.sum(yy_extr * w[:, np.newaxis], axis=0))
res = sp.optimize.least_squares( resto,weights_0 )
wf = res.x
print("Weights for the pristine peaks to get SOBP:")
print(wf)

# plotting
yy = yy/ref # normalize to reference
sobp = np.sum(yy * wf[:, np.newaxis],axis=0)

plt.plot(xx, sobp, lw='0.5')
plt.vlines(a/1e3,0,np.max(yy),linestyles='dashdot',linewidth=0.5,colors="black")
plt.vlines(b/1e3,0,np.max(yy),linestyles='dashdot',linewidth=0.5,colors="black")
plt.hlines(ref/ref,0,5,linestyles='dashdot',linewidth=0.5,colors="red")

yy_w = np.zeros([len(energies), len(xx)])
for i in range(len(energies)):
    yy_w[i,:] = yy[i,:]*wf[i]
for i in range(len(energies)):
    plt.plot(xx, yy_w[i, :], lw='0.5')
plt.title("SOBP")
plt.xlabel("[cm]")
plt.ylabel("Relative dose")
plt.savefig("saved_plots/sobp_1.png",dpi=300)
plt.show()

#OSS: about decreasing last weight: plot again
wf[-1] = wf[-1]*0.1
print("\nWeights but last one is decreased to 10%:")
print(wf)
sobp_2 = np.sum(yy * wf[:, np.newaxis],axis=0)

plt.vlines(a/1e3,0,np.max(yy),linestyles='dashdot',linewidth=0.5,colors="black")
plt.vlines(b/1e3,0,np.max(yy),linestyles='dashdot',linewidth=0.5,colors="black")
plt.plot(xx, sobp_2, lw='0.5')

yy_w_2 = np.zeros([len(energies), len(xx)])
for i in range(len(energies)):
    yy_w_2[i,:] = yy[i,:]*wf[i]
for i in range(len(energies)):
    plt.plot(xx, yy_w_2[i, :], lw='0.5')

plt.title("SOBP + last weight reduced")
plt.xlabel("[cm]")
plt.ylabel("Relative dose")
plt.savefig("saved_plots/sobp_2.png",dpi=300)
plt.show()