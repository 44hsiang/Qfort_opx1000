import matplotlib.pyplot as plt
import numpy as np

def plot_results(title, axes, volume, fitting_error, fidelity, T2, T1, noise_voltage):
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))

    plt.suptitle(title)
    axs_axes =axs[0,0]
    axs_volume =axs[1,0]
    axs_error =axs[0,1]
    axs_fidelity =axs[1,1]
    axs_T2 =axs[2,0]
    axs_T1 =axs[2,1]

    axs_axes.plot(noise_voltage,axes[:,0],'k',label='x')
    axs_axes.plot(noise_voltage,axes[:,1],'b',label='y')
    axs_axes.plot(noise_voltage,axes[:,2],'r',label='z')
    axs_axes.set_ylabel('Axes')
    axs_axes.legend()

    axs_volume.plot(noise_voltage,volume,'k')
    axs_volume.plot(noise_voltage,volume,'.r')
    axs_volume.set_ylabel('Volume')

    axs_error.errorbar(noise_voltage, fitting_error[:,0], yerr=fitting_error[:,1], fmt='o', capsize=5, label="Mean with Std Dev")
    axs_error.set_ylabel('fitting error')

    axs_fidelity.errorbar(noise_voltage, fidelity[:,0], yerr=fidelity[:,1], fmt='o', capsize=5, label="Mean with Std Dev")
    axs_fidelity.set_ylabel('Fidelity')

    axs_T2.plot(noise_voltage,T2,'k')
    axs_T2.plot(noise_voltage,T2,'.r')
    axs_T2.set_xlabel('Noise amp(mV)')
    axs_T2.set_ylabel('T2(us)')

    axs_T1.plot(noise_voltage,T1,'k')
    axs_T1.plot(noise_voltage,T1,'.r')
    axs_T1.set_xlabel('Noise amp(mV)')
    axs_T1.set_ylabel('T1(us)')
    plt.tight_layout()
    plt.show()
    return fig

def plot_avfe(title,xx,axes,volume,fidelity,fitting_error,negativity,T1=False,y_limit=False):
    """
    Plot the average fidelity, fitting error and volume of the ellipsoid
    :param title: title of the plot, string
    :param xx: x axis data, dict{'x_title':np.array(xx)}
    :param data_x: x axis data, np.array
    :param data_y: y axis data, np.array
    :param data_z: z axis data, np.array
    :param volme: volume of the ellipsoid, np.array
    :param fidelity: fidelity of the ellipsoid, np.array
    :param fitting_error: fitting error of the ellipsoid, np.array
    :return: plot
    """
    x_title = list(xx.keys())[0]
    x_values = list(xx.values())[0]
    fig = plt.figure(figsize=(4, 8))
    if type(T1) == np.ndarray or type(T1) == list:
        ax_name = ['axes', 'volume', 'fidelity', 'fitting error', 'negativity','T1']
        ax_T1 = fig.add_subplot(len(ax_name), 1, 6)
        ax_T1.plot(x_values,T1,'k')
    else:
        ax_name = ['axes', 'volume', 'fidelity', 'fitting error', 'negativity']

    ax_axes = fig.add_subplot(len(ax_name), 1, 1)
    ax_volume = fig.add_subplot(len(ax_name), 1, 2)
    ax_fidelity = fig.add_subplot(len(ax_name), 1, 3)
    ax_fitting_error = fig.add_subplot(len(ax_name), 1, 4)
    ax_negativity = fig.add_subplot(len(ax_name), 1, 5)

    
    ax_axes.set_title(title)
    ax_axes.plot(x_values,axes[:,0],'k',label='x')
    ax_axes.plot(x_values,axes[:,1],'r',label='y')
    ax_axes.plot(x_values,axes[:,2],'b',label='z')
    if y_limit:
        ax_axes.set_ylim(0,1)
    ax_axes.set_ylabel('Axes')
    ax_axes.legend(loc='lower right')

    ax_volume.plot(x_values,volume,'k')
    ax_volume.plot(x_values,volume,'.r')
    if y_limit:
        ax_volume.set_ylim(0,4*np.pi/3)
    ax_volume.set_ylabel('Volume')

    ax_fidelity.errorbar(x_values,fidelity[:,0],yerr=fidelity[:,1],fmt='o')
    ax_fidelity.set_ylabel('Fidelity')

    ax_fitting_error.errorbar(x_values,fitting_error[:,0],yerr=fitting_error[:,1],fmt='o')
    ax_fitting_error.set_ylabel('Fitting error')
    #ax_fitting_error.set_xlabel(x_title)

    #ax_negativity.errorbar(x_values,negativity[:,0],yerr=negativity[:,1],fmt='o')
    indices = np.where((negativity > -0.01) & (negativity < 0.5))[0]

    ax_negativity.plot(x_values[indices],negativity[indices],'k')
    ax_negativity.plot(x_values[indices],negativity[indices],'.r')

    ax_negativity.set_ylabel('Negativity')
    if y_limit:
        ax_negativity.set_ylim(0,0.5)
    ax_negativity.set_xlabel(x_title)

    plt.tight_layout()
    return fig
