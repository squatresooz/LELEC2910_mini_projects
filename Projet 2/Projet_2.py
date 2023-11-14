import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Directivity(phi_vec, theta_vec, E):
    dph = phi_vec[1] - phi_vec[0]
    dth = theta_vec[1] - theta_vec[0]
    jc = np.sin(theta_vec) * dth
    jacob = np.outer(jc, np.ones_like(phi_vec)) * dph  # J = sin(θ) * dθ * dϕ en matrice 
    integral = np.sum(np.sum((np.abs(E)**2) * jacob))
    d = (np.abs(E)**2) * 4 * np.pi / integral
    return d

param = np.loadtxt("Antenne Projet 2\code Yagi\int_pat2_R2022a\parameters_500M.txt")
N = int(param[2])
ntheta = 60
nphi = 80

plot = False
PlotDirectivity = True
plot_D_k = True
plot_current = True

Directivity_tab = []

for k in range(1,N+1):
    filename = "Antenne Projet 2\code Yagi\int_pat2_R2022a\pattern{}.txt".format(k)
    data = np.loadtxt(filename)

    j = data[:,0] 
    i = data[:,1] 
    real_tm = data[:,2]
    im_tm = data[:,3]
    theta = np.linspace((np.pi)/(2 * ntheta), ((2 * ntheta + 1) * np.pi)/(2 * ntheta), ntheta, endpoint=True)
    phi = np.linspace(0, (nphi-1)/nphi * 2 * np.pi, nphi,endpoint=True) 

    tm_matrix = real_tm + 1j * im_tm
    tm_matrix = tm_matrix.reshape(len(theta), len(phi))

    #
    # Directivity 
    #
    dd = Directivity(phi, theta, tm_matrix)
    dB = 10 * np.log10(np.max(dd))
    print("Directivite max de l'antenne {} = {:.3f} [dB]\n".format(k,dB)) 
    Directivity_tab.append(dB)

    theta1 = np.outer(np.concatenate(([0], theta)), np.ones_like(np.concatenate((phi, [2*np.pi]))))
    phi1 = np.outer(np.ones_like(np.concatenate(([0], theta))), np.concatenate((phi, [2*np.pi])))
    dd = np.vstack([dd, dd[0,:]])
    dd = np.hstack([dd, dd[:, 0][:, np.newaxis]])

    if PlotDirectivity:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = dd * np.sin(theta1) * np.cos(phi1)
        y = dd * np.sin(theta1) * np.sin(phi1)
        z = dd * np.cos(theta1)

        surf = ax.plot_surface(x, y, z, cmap='plasma')

        ax.view_init(elev=25, azim=-35)    # View from the side
        # ax.view_init(elev=58, azim=-89)   # View from above
        # ax.view_init(elev=55, azim=-73)   # View from above a little to the side

        x_min, x_max = np.floor(x.min()), np.ceil(x.max())
        y_min, y_max = np.floor(y.min()), np.ceil(y.max())
        xy_max = max(x_max, y_max, abs(x_min), abs(y_min))
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_zlim(-5, 5)

        #colorbar = fig.colorbar(surf, location='left')
        #colorbar.set_label("$Directivity$ $[dB]$")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title("Pattern {}".format(k))
        plt.savefig("Antenne Projet 2/Plots/Pattern_{}.pdf".format(k))
        plt.show()

if plot_D_k:
    fig = plt.figure()
    plt.plot(range(1,N+1), Directivity_tab, 'o')
    plt.xlabel("Wire excited [-]")
    plt.ylabel("Directivity [dB]")
    plt.ylim([0,14])
    plt.xticks(range(0,N+2))
    plt.grid()
    # plt.yticks(TF,fontsize=fs_ticks)
    plt.title("Diretivity in terms of the wire excited")
    plt.savefig("Antenne Projet 2/Plots/D_k.pdf")
    plt.show()

#### Current
nb_list = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"]
if plot_current:
    for k in range(1,N+1):
        filename = "Antenne Projet 2\ii\ii{}.txt".format(k)
        data = np.loadtxt(filename)
        reshaped_ii = data.reshape(-1,order='F')
        x = np.linspace(0.5,12.5,len(reshaped_ii),endpoint=True)

        plt.figure()
        plt.plot(x,np.abs(reshaped_ii))
        plt.xlabel("Wire position [-]")
        plt.ylabel("Current distribution [-]")
        plt.xticks(range(0,N+2))
        plt.grid()
        plt.title("Current distribution with the {} wire excited".format(nb_list[k-1]))
        plt.savefig("Antenne Projet 2/Plots/Current_{}.pdf".format(k))
        plt.show()
