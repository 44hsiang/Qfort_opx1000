# %%
from quam_libs.components import QuAM
from quam_libs.quam_builder.machine import build_quam

path = "/Users/4hsiang/Desktop/Jack/python_project/instrument_control/opx1000/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state"

machine = QuAM.load(path)

# octave_settings = {"oct1": {"port": 11235} }  # externally configured: (11XXX where XXX are last three digits of oct ip)
# octave_settings = {"oct1": {"ip": "192.168.88.250"} }  # "internally" configured: use the local ip address of the Octave
octave_settings = {}

# Make the QuAM object and save it
quam = build_quam(machine, quam_state_path=path, octaves_settings=octave_settings)

# %%
