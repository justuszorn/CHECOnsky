from CHECLabPy.core.io import HDF5Reader, HDF5Writer
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
from ctapipe.image import tailcuts_clean, leakage
from ctapipe.image.cleaning import number_of_islands
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.hillas import HillasParameterizationError, \
    hillas_parameters
from CHECLabPy.plotting.camera import CameraImage
from CHECLabPy.utils.mapping import get_ctapipe_camera_geometry
from astropy import units as u
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Ellipse


# TODO:
# - Correct coordinate transformation


def convert_hillas_to_dict(hillas):
    """
    Convert a `HillasParametersContainer` into a dict without the astropy units

    Parameters
    ----------
    hillas : `ctapipe.io.containersHillasParametersContainer`
        The hillas parameters obtained from
        ctapipe.image.hillas.hillas_parameters

    Returns
    -------
    hillas_dict : dict
    """
    hillas_dict = dict()
    for key, value in hillas.items():
        if isinstance(value, u.Quantity):
            value = value.value
        hillas_dict[key] = value
    return hillas_dict


def main():
    description = ('Extract the hillas parameters from a a *_dl1.h5 file to '
                   'produce a *_hillas.h5 file.')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--files', dest='input_paths', nargs='+',
                        help='path to the HDF5 dl1 run files')
    parser.add_argument('-c', '--column', dest='charge_column', action='store',
                        required=True,
                        help='Charge column to use from the DL1 file')
    parser.add_argument('-t', '--tcolumn', dest='time_column', action='store',
                        default=None,
                        help='Time column to use from the DL1 file for '
                             'time gradient calculation')
    parser.add_argument('-n', '--maxevents', dest='max_events', action='store',
                        help='Number of events to process', type=int)
    parser.add_argument('-tc', '--tailcuts', dest='tailcuts', nargs=3,
                        type=int, default=[3, 5, 2],
                        help='Tail cuts used for cleaning (pix pe, neighbour '
                             'pe, min_number_picture_neighbors)')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true',
                        help="Plot images for each event")
    args = parser.parse_args()

    input_paths = args.input_paths
    charge_column = args.charge_column
    time_column = args.time_column
    max_events = args.max_events
    tailcuts = args.tailcuts
    plot = args.plot

    n_files = len(input_paths)

    for ifile, input_path in enumerate(input_paths):
        print("PROGRESS: Processing file {}/{}".format(ifile + 1, n_files))
        output_path = input_path.replace('_dl1.h5', '_hillas.h5')

        with HDF5Reader(input_path) as reader, \
                HDF5Writer(output_path) as writer:
            mapping = reader.get_mapping()
            geom = get_ctapipe_camera_geometry(
                mapping, plate_scale=37.56e-3
            )
            ci = CameraImage.from_mapping(mapping)

            n_events = reader.get_metadata()['n_events']
            n_pixels = reader.get_metadata()['n_pixels']
            run_id = reader.get_metadata()['run_id']
            if max_events and max_events < n_events:
                n_events = max_events

            desc = "Looping over events"
            it = enumerate(reader.iterate_over_chunks('data', n_pixels))
            for ientry, df in tqdm(it, total=n_events, desc=desc):
                iev = df['iev'].values[0]
                t_cpu = df['t_cpu'].values[0]

                if iev >= n_events:
                    break

                # Extract the charge per pixel for the event
                image = df[charge_column].values
                if time_column is not None:
                    image_t = df[time_column].values

                # Clean the image using tailcuts
                tc = tailcuts_clean(
                    geom, image,
                    picture_thresh=tailcuts[0],
                    boundary_thresh=tailcuts[1],
                    min_number_picture_neighbors=tailcuts[2]
                )

                # Get number of islands in image
                num_islands = number_of_islands(geom, tc)

                # Extract the hillas parameters
                try:
                    hillas = hillas_parameters(geom[tc], image[tc])
                except HillasParameterizationError:
                    continue

                # Skip events with a nan width (single pixel width)
                if np.isnan(hillas.width):
                    continue

                # Extract time gradient from cleaned image
                if time_column is not None:
                    image[~tc] = 0
                    image_t[~tc] = 0
                    time_c = timing_parameters(geom, image, image_t, hillas)

                # Get Leakage parameters
                leakage_container = leakage(geom, image, tc)

                # Plot the event if user has requested
                if plot:
                    cleaned_image = np.ma.masked_array(image, mask=~tc)
                    ci.image = cleaned_image
                    ci.highlight_pixels(np.arange(2048), 'black', 0.2, 1)
                    ellipses = ci.ax.findobj(Ellipse)
                    if len(ellipses) > 0:
                        ellipses[0].remove()
                    ellipse = Ellipse(
                        xy=(hillas.x.value, hillas.y.value),
                        width=hillas.length.value,
                        height=hillas.width.value,
                        angle=np.degrees(hillas.psi.rad),
                        fill=False, color='y', lw=2
                    )
                    ci.ax.add_patch(ellipse)
                    plt.pause(1)

                # Store the extracted hillas parameters as a dict
                table = convert_hillas_to_dict(hillas)
                table['nislands'] = num_islands[0]
                if time_column is not None:
                    table['tgradient'] = time_c.slope
                table['leakage1_pixel'] = leakage_container.leakage1_pixel
                table['leakage2_pixel'] = leakage_container.leakage2_pixel
                table['leakage1_intensity'] = leakage_container.leakage1_intensity
                table['leakage2_intensity'] = leakage_container.leakage2_intensity
                table['iobs'] = run_id
                table['iev'] = iev
                table['t_cpu'] = t_cpu

                writer.append(
                    pd.DataFrame(table, index=[ientry]),
                    key='data',
                    expectedrows=n_events
                )

                # Obtain MC information
                index_str = 'index=={}'.format(iev)
                store = reader.store
                if 'mc' in reader.dataframe_keys:
                    index = store.select_as_coordinates('mc', index_str)
                    row = store.select('mc', where=index)
                    row.index = [ientry]
                    row['iobs'] = run_id
                    writer.append(row, key='mc', expectedrows=n_events)
                if 'pointing' in reader.dataframe_keys:
                    index = store.select_as_coordinates('pointing', index_str)
                    row = store.select('pointing', where=index)
                    row.index = [ientry]
                    row['iobs'] = run_id
                    writer.append(row, key='pointing', expectedrows=n_events)

            if 'mc' in reader.dataframe_keys:
                mcheader = reader.get_metadata(key='mc', name='mcheader')
                mcheader['iobs'] = run_id
                df_mch = pd.DataFrame(mcheader, index=[0])
                writer.append(df_mch, key='mcheader', expectedrows=1)

            writer.add_metadata(
                charge_column=charge_column,
                tailcuts=tailcuts,
            )
            writer.add_mapping(mapping)


if __name__ == '__main__':
    main()
