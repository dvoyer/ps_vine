import ps_vine, pickle, argparse
import numpy as np
from os import listdir, makedirs
from os.path import splitext, isfile, join, basename
from PIL import Image

parameters = {
                "data_path": "raw_data/",
                "output_path": "data/",
                "image_path": "textures/",
                "verbose": False,
                "noimg": True
             }

def getImageSetup():
    imagedata = [f for f in listdir(parameters['image_path']) if isfile(join(parameters['image_path'], f))]
    allimages = []
    for fi in imagedata:
        image = Image.open(join(parameters['image_path'], fi))
        try:
            assert image.size[0] == image.size[1]
        except AssertionError:
            print(f"Texture at {join(parameters['image_path'], fi)} is not square.")
            exit()
        try:
            assert image.size[0] % 2 == 0
        except AssertionError:
            print(f"Texture at {join(parameters['image_path'], fi)} does not have an even dimension.")
            exit()
        np_ver = np.asarray(image)
        n_div = 2
        # there's a one line way to do it but i'm tired and don't want to figure it out
        ars = np.hsplit(np_ver, n_div)
        for i in ars:
            allimages += np.vsplit(i, n_div)
    return allimages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ps_prep',
        description='Prepares raw data for use in ps_harmonic'
    )
    #parser.add_argument('files', metavar="file", nargs='+', help='the files to process')
    parser.add_argument('-d', '--indir', nargs='?', 
                        default=parameters['data_path'],
                        help=f"input directory for raw files (default {parameters['data_path']})")
    parser.add_argument('-i', '--img', nargs='?', 
                        default=parameters['image_path'],
                        help=f"path for images (default {parameters['image_path']})")
    parser.add_argument('-n', '--noimg', action='store_false',
                        help=f"does not process images")
    parser.add_argument('-o', '--outdir', nargs='?', 
                        default=parameters['output_path'],
                        help=f"output directory for processed files (default {parameters['output_path']})")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help=f"prints out a lot of info")
    args = parser.parse_args()
    parameters['data_path'] = args.indir
    parameters['image_path'] = args.img
    parameters['output_path'] = args.outdir
    parameters['verbose'] = args.verbose
    parameters['noimg'] = args.noimg
    makedirs(parameters['output_path'], exist_ok=True)
    vine_data = []
    cline_data = []
    if(parameters['verbose']):
        print("Processing frequency data...")
    for i in [f for f in listdir(parameters['data_path']) if isfile(join(parameters['data_path'], f))]:
        try:
            assert splitext(i)[-1].lower() == '.pkl'
        except AssertionError:
            print(f"{i} is not a pickled file, skipping...")
            continue
        try:
            assert "FreqData" in splitext(basename(i))[0]
        except AssertionError:
            print(f"{i} has been renamed, skipping...")
            continue
        with open(join(parameters['data_path'], i), 'rb') as f:
            data = pickle.load(f)
        try:
            assert len(data) == 2
        except AssertionError:
            print(f"{i} does not have the expected structure, skipping...")
            continue
        vine_data.append(data[0])
        cline_data.append(data[1])
    if(parameters['verbose']):
        print(f"Saving outer line data to {parameters['output_path']+'vines.pkl'}...")
    with open(parameters['output_path']+'vines.pkl', "wb+") as f:
        pickle.dump({"stamp": [len(vine_data), ps_vine.fourier_n], "data": vine_data}, f)
    if(parameters['verbose']):
        print(f"Saving center line data to {parameters['output_path']+'clines.pkl'}...")
    with open(parameters['output_path']+'clines.pkl', "wb+") as f:
        pickle.dump({"stamp": [len(cline_data), ps_vine.fourier_n], "data": cline_data}, f)
    freqs = ps_vine.getFreqs()
    if(parameters['verbose']):
        print(f"Saving frequency bucket data to {parameters['output_path']+'freqs.pkl'}...")
    with open(parameters['output_path']+'freqs.pkl', "wb+") as f:
        pickle.dump({"stamp": ps_vine.fourier_n, "data": freqs}, f)
    if parameters['noimg']:
        if(parameters['verbose']):
            print(f"Processing image data...")
        allimages = getImageSetup()
        if(parameters['verbose']):
            print(f"Saving image data to {parameters['output_path']+'allimages.pkl'}...")
        with open(parameters['output_path']+'allimages.pkl', "wb+") as f:
            pickle.dump(allimages, f)
