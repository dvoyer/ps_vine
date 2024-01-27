import ps_vine, pickle, argparse
from os import listdir, makedirs
from os.path import splitext, isfile, basename, isdir, join
from svgpathtools import svg2paths
from math import dist
import numpy as np
from scipy import linalg
from scipy.fft import fft, fftfreq
from random import uniform as r


parameters = {
                "output_path": "raw_data/",
                "verbose": False
             }

def getOffset(t, oline, cline):
    return [oline.point(t).real - cline.point(t).real, oline.point(t).imag - cline.point(t).imag]

def decideCline(paths):
    # assume the longer path is the outer line
    assert len(paths) == 2
    pl0 = paths[0].length()
    pl1 = paths[1].length()
    assert pl0 != pl1
    oline = paths[0] if pl0 < pl1 else paths[1]
    cline = paths[1] if pl0 < pl1 else paths[0]
    #print(pl0 > pl1)
    return oline, cline

def getVineTransform(oline, cline):
    offset_start = getOffset(0, oline, cline)
    offset_end = getOffset(1, oline, cline)
    avg_offset = [0.5*(offset_start[0]+offset_end[0]), 0.5*(offset_start[1]+offset_end[1])]
    oline_pts = ps_vine.parametrizeSection(oline, ps_vine.fourier_n)
    cline_pts = ps_vine.parametrizeSection(cline, ps_vine.fourier_n, offset=avg_offset)
    dist_pts = []
    for i in range(ps_vine.fourier_n):
        dist_pts.append(dist(oline_pts[i], cline_pts[i]))
    yf = fft(dist_pts)
    xf = fftfreq(ps_vine.fourier_n, 1)
    assert ps_vine.magic_frequencies == list(xf)
    return yf

def getClineTransform(cline):
    cline_pts = ps_vine.parametrizeSection(cline, ps_vine.fourier_n, offset=[-1.*cline.point(0).real, -1.*cline.point(0).imag])
    # transform: endpoint goes to 1, 1
    A = np.array([[cline_pts[-1][0], 0],[0, cline_pts[-1][1]]])
    m = linalg.inv(A)
    # transform points
    new_cline_pts = []
    for i in cline_pts:
        b = np.array([[i[0]],[i[1]]])
        new_cline_pts.append([m.dot(b)[0][0], m.dot(b)[1][0]])
    trp = np.transpose(new_cline_pts)
    # new_cline_pts now is boxed from (0, 0) to (1, 1)
    # for each, now find the signed distance from it to y=x
    # THAT's what we get the transform of
    cline_pts = []
    for i in range(len(new_cline_pts)):
        t = i/len(new_cline_pts)
        negFlag = False
        if new_cline_pts[i][1] < t:
            negFlag = True
        cline_pts.append((-1 if negFlag else 1)*dist(new_cline_pts[i], [t, t]))
    yf = fft(cline_pts)
    xf = fftfreq(len(cline_pts), 1)
    assert ps_vine.magic_frequencies == list(xf)
    return yf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ps_deconstruct',
        description='Deconstructs vine data'
    )
    parser.add_argument('files', metavar="file", nargs='+', help='the files to process')
    parser.add_argument('-o', '--outdir', '-d', nargs='?', 
                        default=parameters['output_path'],
                        help=f"output directory for processed files (default {parameters['output_path']})")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help=f"prints out a lot of info")
    args = parser.parse_args()
    parameters['output_path'] = args.outdir
    parameters['verbose'] = args.verbose
    makedirs(parameters['output_path'], exist_ok=True)
    filelist = args.files
    for i in filelist:
        try:
            assert isfile(i)
        except AssertionError:
            print(f"{i} does not exist, skipping...")
            continue
        try:
            assert splitext(i)[-1].lower() == '.svg'
        except AssertionError:
            print(f"{i} is not an svg file, skipping...")
            continue
        if(parameters['verbose']):
            print(i)
        paths, _ = svg2paths(i)
        oline, cline = decideCline(paths)
        oline_freqs = getVineTransform(oline, cline)
        cline_freqs = getClineTransform(cline)
        ofn = f"{parameters['output_path']}FreqData-{splitext(basename(i))[0].lower()}.pkl"
        if(parameters['verbose']):
            print(f"outputting to {ofn}")
        with open(ofn, "wb+") as f:
            pickle.dump([oline_freqs, cline_freqs], f)

    


# with open(f'vines/{filename}.txt', "w+") as f:
#     f.write(str(list(yf)))