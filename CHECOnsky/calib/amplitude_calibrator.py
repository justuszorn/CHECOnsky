from CHECLabPy.calib import AmplitudeCalibrator
from CHECLabPy.utils.files import extract_hv_cfg
import yaml
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyval
from CHECOnsky.calib import get_calib_data
import warnings
from IPython import embed


def get_nudge_from_dacs(dacs):
    dacs_0, hv_0 = extract_hv_cfg(get_calib_data("hv_nudge+0.cfg"))
    dacs_0 = dacs_0.ravel()
    _255 = np.logical_or(dacs == 255, dacs_0 == 255)
    dacs = dacs[~_255]
    dacs_0 = dacs_0[~_255]
    diff = dacs - dacs_0
    if not (diff == diff[0]).all():
        warnings.warn("Not all DACS nudges are equal!", UserWarning)
    nudge = np.median(diff)
    return nudge


def get_nudge_and_temperature_from_reader(reader):
    n_modules = reader.n_modules
    n_sp_per_module = reader.n_superpixels_per_module
    dacs = np.zeros(n_modules * n_sp_per_module)
    sipm_temperature = np.zeros(n_modules)
    for tm in range(n_modules):
        sipm_temperature[tm] = reader.get_sipm_temp(tm)
        for sp in range(n_sp_per_module):
            dacs[tm * n_sp_per_module + sp] = reader.get_sp_dac(tm, sp)
    nudge = get_nudge_from_dacs(dacs)
    temperature = sipm_temperature[sipm_temperature > 0].mean()
    return nudge, temperature


class AstriAmplitudeCalibrator(AmplitudeCalibrator):
    def __init__(self, nudge, temperature, extractor,
                 mv2pe_path=None, temperature_corr_path=None, ff_path=None):
        """
        AmplitudeCalibrator with automatic setup for ASTRI campaign 2019/04

        Parameters
        ----------
        nudge : int
            DAC nudge setting used for observation
        extractor : str
            Name related to the charge extractor used, so that the correct
            coefficients can be retrieved. Should be either
            'charge_cc' or 'charge_averagewf'.
        mv2pe_path : str
            OPTIONAL Specify a different path for the mv2pe conversion
            coefficients. By default, the class uses the default file located
            at CHECOnsky/calib/data/pe_coeff.yml
        ff_path : str
            OPTIONAL Specify a different path for the flat-field
            coefficients. By default, the class uses the default file located
            at CHECOnsky/calib/data/ff.dat
        """
        ff, mv2pe = self.get_coefficients_from_files(
            nudge, extractor, temperature,
            mv2pe_path, temperature_corr_path, ff_path
        )
        super().__init__(ff, np.zeros(ff.shape), mv2pe)  # TODO: Consider pedestal?

    @staticmethod
    def get_coefficients_from_files(
            nudge, extractor, temperature=None,
            mv2pe_path=None, temperature_corr_path=None, ff_path=None
    ):
        if mv2pe_path is None:
            mv2pe_path = get_calib_data("mv2pe.yml")
        if ff_path is None:
            ff_path = get_calib_data("ff_coeff.dat")

        with open(mv2pe_path, 'r') as file:
            mv2pe_dict = yaml.safe_load(file)
        df_ff = pd.read_csv(ff_path, sep='\t')

        mv2pe_coeff = mv2pe_dict[f'{extractor}_coeff']
        mv2pe = polyval(nudge, mv2pe_coeff)
        ff = df_ff[f'{extractor}_ff'].values

        # TODO: Get temperature_correction

        return ff, mv2pe
