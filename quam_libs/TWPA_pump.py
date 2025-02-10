from numpy import *
from RsInstrument import *

def open_TWPA(addr,power,pump_frequency,gain,show_parameter=False):
    power = True
    TWPA_LO = pump_frequency
    instr = RsInstrument(addr, True, True)
# 2. instrument setting parameter
    instr.visa_timeout = 1000  # milliseconds
    instr.instrument_status_checking = True
    instr.write_bool('OUTPUT', power)  # sending 'SOURCE:RF:OUTPUT:STATE ON'
    instr.write_float('POWER', gain)
    instr.write_float('FREQUENCY', TWPA_LO)  # sending 'SOURCE:RF:FREQUENCY 1000000000'
    instr.write_bool('PULM:STATE', False)  # sending 'SOURCE:RF:OUTPUT:STATE ON'
    instr.write_str('ROSC:SOUR INT')
#instr.write_str('ROSC:EXT:FREQ 1000MHZ')
#instr.write_str('ROSC:OUTP:FREQ 1000MHZ')
    instr.write_str('CONN:REFL:OUTP REF')
    instr.write_bool('IQ:STATe', False)

# instrument setting parameter check
    if show_parameter:
        RF_output = instr.query_bool('OUTPUT?')  # returning boolean out=True
        power = instr.query_float('POWER?')  # returning boolean out=True
        freq = instr.query_float('FREQUENCY?')  # returning float number freq=1E9
        pulm = instr.query_bool('PULM:STATE?')  # sending 'SOURCE:RF:OUTPUT:STATE ON'
        LO_source = instr.query_str('LOSC:SOUR?')
        Ref_source = instr.query_str('ROSC:SOUR?')
        Ref_input_freq = instr.query_str('ROSC:EXT:FREQ?')
        IQ = instr.query_bool('IQ:STAT?')

        print('=======TWPA setting========')
        print(f'TWPA pump = {RF_output}')
        print(f'drive power = {str(power)} dBm')
        print(f'Frequency = {str(round(freq / 1e9, 4))} GHZ')
        print(f'Pulse mode = {pulm}')
        print(f'LO source = {LO_source}')
        print(f'Ref source = {Ref_source} and input frequency = {Ref_input_freq}')
        print(f'IQ modulation = {IQ}')

# Close the session
    #instr.close()
def TWPA_info(addr):
    instr = RsInstrument(addr, False, False)
    RF_output = instr.query_bool('OUTPUT?')  # returning boolean out=True
    power = instr.query_float('POWER?')  # returning boolean out=True
    freq = instr.query_float('FREQUENCY?')  # returning float number freq=1E9
    return RF_output, power, freq