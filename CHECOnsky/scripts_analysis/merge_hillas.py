from CHECLabPy.core.io import HDF5Reader, HDF5Writer
from CHECLabPy.utils.files import sort_file_list
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter


def main():
    description = ('Merge together the hillas.h5 files into a single '
                   'hillas.h5 file.')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--files', dest='input_paths', nargs='+',
                        help='paths to the HDF5 hillas files')
    parser.add_argument('-o', dest='output_path', required=True,
                        help='output path to store the merged file')
    args = parser.parse_args()

    input_paths = sort_file_list(args.input_paths)
    output_path = args.output_path

    n_files = len(input_paths)

    with HDF5Writer(output_path) as writer:
        for ifile, input_path in enumerate(input_paths):
            print("PROGRESS: Processing file {}/{}".format(ifile + 1, n_files))

            with HDF5Reader(input_path) as reader:
                if ifile == 0:
                    writer.add_mapping(reader.get_mapping())
                    writer.add_metadata(**reader.get_metadata())

                keys = ['data', 'pointing', 'mc', 'mcheader']
                for key in keys:
                    if key not in reader.dataframe_keys:
                        continue

                    it = enumerate(reader.iterate_over_chunks(key, 1000))
                    for ientry, df in it:
                        writer.append(df, key=key)


if __name__ == '__main__':
    main()
