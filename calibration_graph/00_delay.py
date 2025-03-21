# %%
from qm.qua import *
from qualang_tools.units import unit
from quam_libs.components import QuAM
import matplotlib.pyplot as plt


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("/Users/4hsiang/Desktop/Jack/python_project/instrument_control/opx1000/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state")
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
qmm = machine.connect()

qubits = machine.active_qubits

with program() as prog:

    with infinite_loop_():

        qubits[0].xy.play('saturation')
        qubits[-1].xy.play('saturation')
        #qubits[0].z.play('const')
        qubits[0].z.play('const')
        qubits[1].resonator.play('readout')
        wait(20000)
        align()

qm = qmm.open_qm(config)
job = qm.execute(prog)
plt.show()


# %%
