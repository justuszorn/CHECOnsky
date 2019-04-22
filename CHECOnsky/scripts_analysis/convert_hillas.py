from CHECLabPy.core.io import HDF5Reader

import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter

from fact.io import to_h5py

import pandas as pd

import astropy.units as u
import h5py

def main():
    description = ('Convert hillas file to h5py.')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--files', dest='input_path',
                        help='path to the HDF5 hillas files')
    parser.add_argument('-o', dest='output_path', required=True,
                        help='output path to store the h5py file')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    with HDF5Reader(input_path) as reader:
        tel = reader.read('data')
        arr = reader.read('mc')
        pnt = reader.read('pointing')
        run = reader.read('mcheader')

    arr = arr.rename(columns={
                      'energy':'mc_energy', 
                      'alt':'mc_alt',
                      'az':'mc_az',
                      'core_x':'mc_core_x',
                      'core_y':'mc_core_y',
                      'h_first_int':'mc_h_first_int',
                      'shower_primary_id':'mc_shower_primary_id',
                      'x_max':'mc_x_max',
                      #'iobs':'run_id',
                      #'iev': 'array_event_id'
                      })
    pnt = pnt.drop(columns=['t_cpu'])
    arr = pd.merge(arr, pnt, on=['iobs','iev'])
    arr = arr.rename(columns={
                      'iobs':'run_id',
                      'iev': 'array_event_id'
                      })

    tel['array_event_id'] = tel.iev.values
    tel = tel.rename(columns={'iev':'telescope_event_id',
                              'iobs': 'run_id'})
    tel = tel.drop(columns=['t_cpu'])
    plate_scale = 37.56
    tel.x = tel.x * plate_scale
    tel.y = tel.y * plate_scale

    run = run.rename(columns={'iobs':'run_id'})

    to_h5py(tel, output_path, key='telescope_events', mode='w')
    to_h5py(arr, output_path, key='array_events', mode='a')
    to_h5py(run, output_path, key='runs', mode='a')


if __name__ == '__main__':
    main()
