#%%
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.units import unit
from quam_libs.components import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from qualang_tools.units import unit
from quam_libs.components import QuAM

#%%
###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
path = "/Users/4hsiang/Desktop/Jack/python_project/instrument_control/opx1000/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state"
machine = QuAM.load(path)
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

qubits = machine.active_qubits
%matplotlib qt
simulate = False
with program() as prog:

    #qubits[2].xy.update_frequency(-100e6)
    #qubits[1].xy.update_frequency(-100e6)
    #qubits[0].xy.update_frequency(-100e6)

    a = declare(fixed)

    with infinite_loop_():

        # with for_(a, 0, a < 1.0, a +0.1):

        #qubits[0].xy.play("x180")
        # qubits[1].xy.play('x180')
        # wait(4)
        # align()
        #align()
        #qubits[0].resonator.play("const")
        #align()
        qubits[0].z.play("const")
        #wait(100)

        # for qubit in machine.active_qubits:
        #     qubit.z.play('const')
        #     qubit.z.wait(4)

        # with for_(a, 0, a < 2.0, a+0.2):
        # # for qubit in machine.active_qubits:
        #     qubits[2].xy.play('x180', amplitude_scale=a)
        #     qubits[2].xy.wait(4)


if simulate:
    job = qmm.simulate(config, prog, SimulationConfig(duration=1000))
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(prog)

plt.show()



# %%
