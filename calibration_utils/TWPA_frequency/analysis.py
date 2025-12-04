import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""

    frequency: float
    fwhm: float
    success: bool
    best_twpa_freq: float | None = None
    best_snr: float | None = None


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_qubit = f"Results for qubit {q}: "
        s_freq = f"\tResonator frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
        s_freq_twpa = (
            f"\tBest TWPA frequency: {1e-6 * fit_results[q]['best_twpa_freq']:.2f} MHz | "
            if fit_results[q]["best_twpa_freq"] is not None
            else "\tBest TWPA frequency: N/A | "
        )
        s_snr = f"Best SNR: {fit_results[q]['best_snr']:.2f}\n"
        log_callable(s_qubit + s_freq + s_fwhm + s_freq_twpa + s_snr)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    # Fit the resonator line
    fit_results = peaks_dips(ds.IQ_abs, "detuning")
    # Compute SNR vs TWPA pump frequency (mean/std over detuning)
    snr, best_twpa_freq, best_snr = _compute_snr_vs_twpa_freq(ds)
    # Extract the relevant fitted parameters (and optionally best TWPA info)
    fit_data, fit_results = _extract_relevant_fit_parameters(
        fit_results, node, best_twpa=best_twpa_freq, best_snr=best_snr
    )
    # Attach SNR info to dataset and results
    if snr is not None:
        fit_data = fit_data.assign(snr=snr)
        fit_data = fit_data.assign_coords(
            best_twpa_freq=("qubit", best_twpa_freq.data),
            best_snr=("qubit", best_snr.data),
        )
        fit_data.best_twpa_freq.attrs = {"long_name": "best TWPA frequency", "units": "Hz"}
        fit_data.best_snr.attrs = {"long_name": "best SNR", "units": "mean/std"}
        node.results["snr_vs_twpa_freq"] = snr
        node.results["best_twpa_freq"] = {q: best_twpa_freq.sel(qubit=q).item() for q in best_twpa_freq.qubit.values}
        node.results["best_snr"] = {q: best_snr.sel(qubit=q).item() for q in best_snr.qubit.values}
    return fit_data, fit_results


def _extract_relevant_fit_parameters(
    fit: xr.Dataset,
    node: QualibrationNode,
    best_twpa: xr.DataArray | None = None,
    best_snr: xr.DataArray | None = None,
):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}
    # Get the fitted resonator frequency
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    full_freq_da = xr.DataArray(full_freq, dims=["qubit"])
    res_freq = fit.position + full_freq_da  # broadcast qubit baseline over TWPA_freq if present
    # Align/broadcast resonator frequency across TWPA_freq while tied to each qubit
    res_freq_coord = res_freq
    dim_twpa = "TWPA_freq"
    if dim_twpa in fit.dims and dim_twpa not in res_freq_coord.dims:
        res_freq_coord = res_freq_coord.expand_dims(**{dim_twpa: fit.coords[dim_twpa]})
    fit = fit.assign_coords(res_freq=res_freq_coord)
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=fwhm)
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}
    # Assess whether the fit was successful or not
    freq_success = np.abs(res_freq_coord) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq_da
    fwhm_success = np.abs(fwhm) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq_da
    success_criteria = freq_success & fwhm_success
    fit = fit.assign_coords(success=success_criteria)

    fit_results = {}
    for q in fit.qubit.values:
        fit_results[q] = FitParameters(
            frequency=_scalar_from_qubit_coord(fit.sel(qubit=q).res_freq),
            fwhm=_scalar_from_qubit_coord(fit.sel(qubit=q).fwhm),
            success=bool(_scalar_from_qubit_coord(fit.sel(qubit=q).success)),
            best_twpa_freq=float(best_twpa.sel(qubit=q)) if best_twpa is not None else None,
            best_snr=float(best_snr.sel(qubit=q)) if best_snr is not None else None,
        )
    return fit, fit_results


def _scalar_from_qubit_coord(da: xr.DataArray):
    """Extract a single scalar from a qubit-indexed DataArray that may include TWPA_freq."""
    if "TWPA_freq" in da.dims:
        da = da.isel(TWPA_freq=0)
    return da.values.item()


def _compute_snr_vs_twpa_freq(ds: xr.Dataset) -> Tuple[xr.DataArray | None, xr.DataArray | None, xr.DataArray | None]:
    """
    Estimate SNR per TWPA pump frequency for each qubit using IQ_abs along detuning.

    SNR is defined as mean(IQ_abs) / std(IQ_abs) over detuning for each TWPA frequency.
    Returns Nones if TWPA_freq is not present.
    """
    dim = "TWPA_freq"
    if dim not in ds.dims:
        return None, None, None
    iq = ds.IQ_abs
    signal = iq.mean(dim="detuning")
    noise = iq.std(dim="detuning")
    snr = (signal / noise).rename("snr")
    best_twpa = snr.idxmax(dim=dim)
    best_snr = snr.max(dim=dim)
    return snr, best_twpa, best_snr
