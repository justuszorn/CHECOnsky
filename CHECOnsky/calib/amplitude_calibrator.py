from CHECLabPy.calib import AmplitudeCalibrator
from CHECLabPy.utils.files import extract_hv_cfg
import yaml
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyval
from CHECOnsky.calib import get_calib_data
import warnings


def get_nudge_from_dacs(dacs):
    """
    Calculate the nudge from the superpixel DAC values

    Parameters
    ----------
    dacs : ndarray
        1D array of dac values of size n_superpixels (512)

    Returns
    -------
    nudge : int
        The median nudge value for the camera
    """
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
    """
    Obtain the nudge and average camera temperature from the header of a file

    Parameters
    ----------
    reader : CHECLabPy.core.io.TIOReader or CHECLabPy.core.io.DL1Reader
        Reader of the file

    Returns
    -------
    nudge : float
    temperature : float
    """
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
                 mv2pe_path=None, temperature_path=None, ff_path=None):
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
        temperature : float
            Average camera temperature
        mv2pe_path : str
            OPTIONAL Specify a different path for the mv2pe conversion
            coefficients. By default, the class uses the default file located
            at CHECOnsky/calib/data/pe_coeff.yml
        temperature_path : str
            OPTIONAL Specify a different path for the temperature correction.
            By default, the class uses the default file located
            at CHECOnsky/calib/data/temperature_coeff.yml
        ff_path : str
            OPTIONAL Specify a different path for the flat-field
            coefficients. By default, the class uses the default file located
            at CHECOnsky/calib/data/ff.dat
        """
        self.extractor = extractor
        ff, mv2pe = self.get_coefficients_from_files(
            nudge, extractor, temperature,
            mv2pe_path, temperature_path, ff_path
        )
        super().__init__(ff, np.zeros(ff.shape), mv2pe)
        # TODO: Consider pedestal?

    @staticmethod
    def get_coefficients_from_files(
            nudge, extractor, temperature=None,
            mv2pe_path=None, temperature_path=None, ff_path=None
    ):
        """

        Parameters
        ----------
        nudge : int
            DAC nudge setting used for observation
        extractor : str
            Name related to the charge extractor used, so that the correct
            coefficients can be retrieved. Should be either
            'charge_cc' or 'charge_averagewf'.
        temperature : float
            Average camera temperature
        mv2pe_path : str
            OPTIONAL Specify a different path for the mv2pe conversion
            coefficients. By default, the class uses the default file located
            at CHECOnsky/calib/data/pe_coeff.yml
        temperature_path : str
            OPTIONAL Specify a different path for the temperature correction.
            By default, the class uses the default file located
            at CHECOnsky/calib/data/temperature_coeff.yml
        ff_path : str
            OPTIONAL Specify a different path for the flat-field
            coefficients. By default, the class uses the default file located
            at CHECOnsky/calib/data/ff.dat

        Returns
        -------
        ff : ndarray
        mv2pe : float

        """
        if mv2pe_path is None:
            mv2pe_path = get_calib_data("mv2pe.yml")
        if ff_path is None:
            ff_path = get_calib_data("ff_coeff.dat")
        if temperature_path is None:
            temperature_path = get_calib_data("temperature_coeff.yml")

        df_ff = pd.read_csv(ff_path, sep='\t')
        with open(mv2pe_path, 'r') as file:
            mv2pe_dict = yaml.safe_load(file)
        with open(temperature_path, 'r') as file:
            temperature_dict = yaml.safe_load(file)

        ff = df_ff[f'{extractor}_ff'].values

        mv2pe_coeff = mv2pe_dict[f'{extractor}_coeff']
        nudge_max = mv2pe_dict['nudge_max']
        nudge_min = mv2pe_dict['nudge_min']
        if not nudge_min <= nudge <= nudge_max:
            raise ValueError(
                "Nudge is not within range: "
                f"{nudge_min} <= {nudge} <= {nudge_max}"
            )
        mv2pe = polyval(nudge, mv2pe_coeff)

        temperature_coeff = temperature_dict[f'temperature_coeff']
        temp_max = temperature_dict['temperature_max']
        temp_min = temperature_dict['temperature_min']
        if not temp_min <= temperature <= temp_max:
            raise ValueError(
                "Temperature is not within range: "
                f"{temp_min} <= {temperature} <= {temp_max}"
            )
        temperature_corr = polyval(temperature, temperature_coeff)
        mv2pe *= temperature_corr

        return ff, mv2pe

    def include_ff_correction(self, ff_correction_path):
        """
        Include the correction factors to the flat-field coefficients
        calculated from LED flasher runs and processed with
        CHECOnsky/scripts_analysis/calculate_ff_correction.py.

        This corrects for any changes to the FF coefficients due to changes
        in HV or NSB.

        Parameters
        ----------
        ff_correction_path : str
            Path to the file created with calculate_ff_correction.py
        """
        df_ff = pd.read_csv(ff_correction_path, sep='\t')
        ff = df_ff[f'{self.extractor}_ff'].values
        self.ff *= ff
