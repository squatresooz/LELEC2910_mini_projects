import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def Directivity_CO_CROSS(phi_vec, theta_vec, CO_pol, CROSS_pol):
    Etot = np.abs(CO_pol)**2 + np.abs(CROSS_pol)**2

    dph = phi_vec[1] - phi_vec[0]
    dth = theta_vec[1] - theta_vec[0]
    jc = np.sin(theta_vec) * dth
    jacob = np.outer(jc, np.ones_like(phi_vec)) * dph  # J = sin(θ) * dθ * dϕ en matrice 
    integral = np.sum(np.sum(Etot * jacob))
    dco = 10*np.log10(np.abs(CO_pol)**2 * 2 * np.pi / integral)
    dcross = 10*np.log10(np.abs(CROSS_pol)**2 * 2 * np.pi / integral)
    return dco, dcross

def Directivity_theta_phi(phi_vec, theta_vec, E_theta_phi, Etot):
    dph = phi_vec[1] - phi_vec[0]
    dth = theta_vec[1] - theta_vec[0]
    jc = np.sin(theta_vec) * dth
    jacob = np.outer(jc, np.ones_like(phi_vec)) * dph  # J = sin(θ) * dθ * dϕ en matrice 
    integral = np.sum(np.sum(Etot * jacob))
    d = np.abs(E_theta_phi) * 2 * np.pi / integral
    return d

phi_mat = scipy.io.loadmat('Antenne Projet 1/Metasurface patterns/phi_grid_matrix.mat')
theta_mat = scipy.io.loadmat('Antenne Projet 1/Metasurface patterns/Theta_grid_matrix.mat')
S12_TE_mat = scipy.io.loadmat('Antenne Projet 1/Metasurface patterns/S12_grid_matrix_TE.mat')
S12_TM_mat = scipy.io.loadmat('Antenne Projet 1/Metasurface patterns/S12_grid_matrix_TM.mat')

phi = np.array(phi_mat['phi_grid_matrix'])
theta = np.array(theta_mat['Theta_grid_matrix'])
S12_TE = np.array(S12_TE_mat['S12_grid_matrix_TE'])
S12_TM = np.array(S12_TM_mat['S12_grid_matrix_TM'])

theta_vec = theta[:,0]
phi_vec = phi[0,:]

E2 = np.abs(S12_TM)**2 + np.abs(S12_TE)**2

########################### Q1 ############################################
max_index = np.unravel_index(np.argmax(E2, axis=None), E2.shape)
print("Index of maximum value:", max_index)
print("Max at theta = %.1f°" % (theta_vec[max_index[0]] * 180/np.pi), ", phi = %.1f°\n" % (phi_vec[max_index[1]] * 180/np.pi))

LHCP_coeff = np.zeros_like(S12_TM)
RHCP_coeff = np.zeros_like(S12_TM)
LHCP_coeff = (S12_TM + S12_TE * 1j) / np.sqrt(2) # e_{TM}  -> e_{\theta} <=> e_x' and LHCP = e_x' + 1j*e_y' => E = TM e_x' - TE e_y'
RHCP_coeff = (S12_TM - S12_TE * 1j) / np.sqrt(2) # -e_{TE} -> e_{\phi}   <=> e_y' and RHCP = e_x' - 1j*e_y'      = \alpha * LHCP + \beta * RHCP

print("############# Q1 #############")
print("absolue LHCP coeff: alpha = %.4f" % np.abs(LHCP_coeff[max_index[0],max_index[1]]), " / absolue RHCP coeff: beta = %.4f" % np.abs(RHCP_coeff[max_index[0],max_index[1]]))
if (np.abs(LHCP_coeff[max_index[0],max_index[1]]) > np.abs(RHCP_coeff[max_index[0],max_index[1]])):
    print("The polarisation is Left Hand Circular Polarisation\n")
    CO_pol = LHCP_coeff
    CROSS_pol = RHCP_coeff
else:
    print("The polarisation is Right Hand Circular Polarisation\n")
    CO_pol = RHCP_coeff
    CROSS_pol = LHCP_coeff

############################# Q2 ###########################################
Directivity_plot = True
pol_plots = True
pol_plots_directivity = True
circle_coords = (np.sin(theta_vec[max_index[0]])*np.cos(phi_vec[max_index[1]]), np.sin(theta_vec[max_index[0]])*np.sin(phi_vec[max_index[1]]))
u = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
dco, dcross = Directivity_CO_CROSS(phi_vec, theta_vec, CO_pol, CROSS_pol)

if Directivity_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf =  ax.plot_surface(
        dco * np.sin(theta) * np.cos(phi),
        dco * np.sin(theta) * np.sin(phi),
        dco * np.cos(theta),
        cmap='plasma'
    )
    colorbar = fig.colorbar(surf, location='left')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("Directivity Co-pol")
    plt.savefig("Antenne Projet 1/Plots/CO_poll_directivity_xyz.pdf")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf =  ax.plot_surface(
        dcross * np.sin(theta) * np.cos(phi),
        dcross * np.sin(theta) * np.sin(phi),
        dcross * np.cos(theta),
        cmap='plasma'
    )
    colorbar = fig.colorbar(surf, location='left')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("Directivity Cross-pol")
    plt.savefig("Antenne Projet 1/Plots/CROSS_poll_directivity_xyz.pdf")
    plt.show()

if pol_plots:
    fig, ax = plt.subplots()
    plt.title("Co-pol")
    contour = ax.contourf(u[0], u[1], np.abs(CO_pol)*10**3, cmap='plasma', levels=100)
    ax.set_ylabel('v')
    ax.set_xlabel('u')
    cbar = plt.colorbar(contour)
    cbar.set_label("$\\alpha$ $[\\frac{mV}{m}]$")
    ax.scatter(*circle_coords, color='none', edgecolor='red', linewidth=1.5, label = "maximum Electric field")
    ax.legend()
    plt.savefig("Antenne Projet 1/Plots/CO_poll.pdf")
    plt.show()

    fig, ax = plt.subplots()
    plt.title("Cross-pol")
    contour = ax.contourf(u[0], u[1], np.abs(CROSS_pol)*10**3, cmap='plasma', levels=100)
    ax.set_ylabel('v')
    ax.set_xlabel('u')
    cbar = plt.colorbar(contour)
    cbar.set_label("$\\beta$ $[\\frac{mV}{m}]$")
    ax.scatter(*circle_coords, color='none', edgecolor='red', linewidth=1.5, label = "maximum Electric field")
    ax.legend()
    plt.savefig("Antenne Projet 1/Plots/CROSS_poll.pdf")
    plt.show()

if pol_plots_directivity:
    fig, ax = plt.subplots()
    plt.title("Co-pol Directivity")
    contour = ax.contourf(u[0], u[1], dco, cmap='plasma', levels=100, vmin=-40, vmax=20)
    ax.set_ylabel('v')
    ax.set_xlabel('u')
    cbar = plt.colorbar(contour)
    cbar.set_label("$D_{Co\\_ pol}$ $[dB]$")
    ax.scatter(*circle_coords, color='none', edgecolor='green', linewidth=1.5, label = "maximum Electric field")
    ax.legend()
    plt.savefig("Antenne Projet 1/Plots/CO_poll_directivity.pdf")
    plt.show()

    fig, ax = plt.subplots()
    plt.title("Cross-pol Directivity")
    contour = ax.contourf(u[0], u[1], dcross, cmap='plasma', levels=100, vmin=-40, vmax=20)
    ax.set_ylabel('v')
    ax.set_xlabel('u')
    cbar = plt.colorbar(contour)
    cbar.set_label("$D_{Cross\\_ pol}$ $[dB]$")
    ax.scatter(*circle_coords, color='none', edgecolor='green', linewidth=1.5, label = "maximum Electric field")
    ax.legend()
    plt.savefig("Antenne Projet 1/Plots/CROSS_poll_directivity.pdf")
    plt.show()

############################ Q3 ########################################
theta_index = np.where(theta_vec == 30*np.pi/180)[0][0]
phi_index = np.where(phi_vec == 90*np.pi/180)[0][0]

# Given values
transmitted_power = 1.0 * 10**(-3)  # Transmitted power in W
gain_Yagi_dB = 10.0  # Gain of the Yagi antenna in dB
gain_Yagi = 10**(gain_Yagi_dB/10)
frequency_GHz = 24.0  # Frequency in GHz
distance_m = 100.0  # Distance between antennas in meters
radiation_efficiency = 0.80  # Radiation efficiency
speed_of_light = 3e8  # Speed of light in meters per second
wavelength_m = speed_of_light / (frequency_GHz * 1e9)  # Convert GHz to Hz
directivity = Directivity_theta_phi(phi_vec, theta_vec, E2[theta_index][phi_index], E2)
reflection_coeff = 0  # Reflection coefficient
polarization_factor = np.sqrt((np.abs(S12_TM[theta_index][phi_index])**2)/(E2[theta_index][phi_index]))  # Polarization factor (e_theta)/(e_theta + e_phi) = (TM)/(TM+TE)

# Calculate the received power with the updated formula
received_power_W = (transmitted_power * (radiation_efficiency ** 2) *  gain_Yagi * directivity * (wavelength_m ** 2)) / ((4 * np.pi)**2 * distance_m ** 2) * (1 - abs(reflection_coeff) ** 2) * polarization_factor
# Convert received power to dBm (decibel-milliwatts)
received_power_dBm = 10 * np.log10(received_power_W * 10**3)

print("############# Q3 #############")
print("Received Power: %.3f" % (received_power_W*10**(12)), " [pW] or %.3f" % (received_power_dBm), " [dBm]\n")


#PS entre pola réellement envoyée mais normalisée + pola de l'antenne réceptrice
#changement de base avec base orthornormée ! => projection possible alors

plot = False
if plot :
    # =============================================================================
    # S12_TE
    # =============================================================================
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("TE")
    surf = ax.plot_surface(phi, theta, S12_TE, cmap='plasma',linewidth=0, antialiased=False)
    
    ax.set_ylabel('$\\theta$')
    ax.set_xlabel('$\\phi$')
    ax.set_zlabel('S12_TE')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    
    # =============================================================================
    # S12_TM
    # =============================================================================
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("TM")
    surf = ax.plot_surface(phi, theta, S12_TM, cmap='plasma',linewidth=0, antialiased=False)
    ax.set_ylabel('$\\theta$')
    ax.set_xlabel('$\\phi$')
    ax.set_zlabel('S12_TM')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    # =============================================================================
    # Plot E  
    # =============================================================================
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("E")
    surf = ax.plot_surface(phi, theta, E2, cmap='plasma',linewidth=0, antialiased=False)
    ax.set_ylabel('$\\theta$')
    ax.set_xlabel('$\\phi$')
    ax.set_zlabel('$|E|^2$')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
