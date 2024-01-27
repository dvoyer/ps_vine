import random, pickle, traceback, argparse
import ps_vine
import numpy as np
from svgpathtools import Path, Line, svg2paths2, wsvg
from PIL import Image, ImageFilter
from os import makedirs
from os.path import splitext, isfile, basename, isdir, join

# data paths
data = {
        "images": "data/allimages.pkl",
        "clines": "data/clines.pkl",
        "olines": "data/vines.pkl",
        "freqs": "data/freqs.pkl"
       }

debug_opts = {
              "expert":                False, # expert mode
              "log_SI_check_too_many": False, # check issues with self-intersection code (too many intersections)
              "log_SI_check_points":   False, # check issues with self-intersection code (log the intersection points)
              "log_SI_SVG_check":      True, # check self-intersection via svg
              "log_pos_neg":           False, # check the actual pos and neg paths via SVG
              "bypass_SI_check":       False, # bypass self-intersection code. DO NOT USE.
              "output_pngs":           True, # output PNGs. DO NOT CHANGE.
              "output_svgs":           False  # also output SVGs of vines
             }

parameters = {
              "verbose": False,                   # verbose mode
              "output_path": "output/",           # output file path
              "glob_n": 2**10,                    # global number of sample points, not used in fourier construction
              "wsize": 32,                        # window size for envelope
              "middle_freq_scale": 0.15,          # scale middle frequencies by this amount
              "amp_filter": [                     # remove freqs where abs(avg) < i[0] and std < i[1]
                             [0.33, 0.5],         
                             [0.25, 0.75]
                            ],
              "sigma": 0.75,                      # value for blur
              "vineVarMin": 0.85,                 # maximum variance for vine distance
              "vineVarMax": 1.1,                  # minimum variance for vine distance
              "enforce_minimum_separation": True, # force a minimum interior size
              "minimum_separation_value": 0.05    # the value to enforce
             }

def SVGtopts(inp: Path, n=parameters["glob_n"]):
    return [inp.point(i/n) for i in range(n)]

def setSeed(sd=-1):
    # seed = 2494458808 # big crossover
    # seed = 1344364672912559207 # window test
    # seed = 1117745755340482043 # crossover and scaling testing
    # seed = 8754881419683577764
    # seed = 3883827076331450744
    #hack to get seed
    if sd == -1:
        seed = random.randint(1,2**64)
    else:
        seed = sd
    print(f"Seed: {seed}")
    random.seed(seed)

def simplify(coeffs, freqs):
    simplified_coeffs = []
    for i in range(len(coeffs)):
        avg = np.average(coeffs[i])
        std = np.std(coeffs[i])
        for j in parameters["amp_filter"]:
            if (abs(avg) < j[0] and std < j[1]):
                pass#print("(0) 0")
        else:
            simplified_coeffs.append([freqs[i], coeffs[i]])
            # print(f"{freqs[i]} - {avg} ({std})")
    return simplified_coeffs

def getSetup():
    with open(data["olines"], "rb") as f:
        vines = pickle.load(f)
    with open(data["clines"], "rb") as f:
        cline = pickle.load(f)
    with open(data["freqs"], "rb") as f:
        freqs = pickle.load(f)
    vine_data = vines["data"]
    cline_data = cline["data"]
    vine_len = len(vine_data)
    cline_len = len(cline_data)
    vine_stamp = vines["stamp"]
    cline_stamp = cline["stamp"]
    freq_stamp = freqs["stamp"]
    try:
        assert vine_stamp[1] == ps_vine.fourier_n
        assert cline_stamp[1] == ps_vine.fourier_n
        assert freq_stamp == ps_vine.fourier_n
    except AssertionError:
        print("ERROR: Mismatched fourier frequencies. Recompile the data with ps_prep_data.py")
        traceback.print_exc()
        exit()
    try:
        assert vine_stamp[0] == vine_len
        assert cline_stamp[0] == cline_len
        assert vine_len == cline_len
    except AssertionError:
        print("ERROR: Data files are malformed. Recompile the data with ps_prep_data.py")
        traceback.print_exc()
        exit()
    vine_coeffs = simplify(np.array(vine_data).transpose(), freqs["data"])
    cline_coeffs = simplify(np.array(cline_data).transpose(), freqs["data"])
    with open(data["images"], "rb") as f:
        allimages = pickle.load(f)
    return vine_coeffs, cline_coeffs, allimages

def getForm(simplified_coeffs):
    actual_coeffs = []
    # create the transform
    for i in range(len(simplified_coeffs)):
        actual_coeffs.append([simplified_coeffs[i][0], random.choice(simplified_coeffs[i][1])])
    # magic
    t = np.linspace(0.0, 2**10, 2**10, endpoint=False)
    y = np.zeros_like(t, dtype=np.complex128)
    radangle = 0.5*np.pi
    ncoeff = len(actual_coeffs)
    bands = [0.5*x for x in [.3, .7]]
    for j in range(ncoeff):
        i = actual_coeffs[j]
        if (bands[0]*ncoeff < j and j < bands[1]*ncoeff) or ((0.5+bands[0])*ncoeff < j and j < (0.5+bands[1])*ncoeff):
            fudge = parameters["middle_freq_scale"]
        else:
            fudge = 1
        jitter = fudge * np.exp(random.uniform(-1*radangle, radangle)*1j)
        y += jitter * i[1] * np.exp(1j * (2 * np.pi * i[0] * t))

    return y.real#/y.real.max()

def envelope(f):
    fc = f
    l = len(f)
    marr = [np.exp(-1./(4.*(i+0.001)/parameters["wsize"])) for i in range(parameters["wsize"])]
    for i in range(parameters["wsize"]):
        fc[i] *= marr[i]
        fc[l-i-1] *= marr[i]
    return fc

def rescale(pos, neg, log=False, max_var=1):
    posmin = min(pos)
    negmax = max(neg)
    # calculate the new zero point
    nz = (pos[0] - posmin + neg[0] - negmax)*0.5
    pos = [i + nz for i in pos]
    neg = [i + nz for i in neg]
    # recalc
    posmin = min(pos)
    negmax = max(neg)
    diff = [pos[i] - neg[i] for i in range(len(pos))]
    if max_var == 1:
        rmdiff = 1/max(diff)
    elif max_var > 1:
        rmdiff = random.randint(1, int(max_var))/max(diff)
    else:
        raise ValueError("invalid max variation")
    mdi = diff.index(max(diff))
    if log:
        print(mdi, max(diff), pos[mdi], neg[mdi], pos[mdi]*rmdiff, neg[mdi]*rmdiff)
        print(posmin, pos.index(posmin))
    rescaleFactor = 1.0
    pos = [(i - posmin)*rmdiff + rescaleFactor*posmin*rmdiff for i in pos]
    neg = [(i - negmax)*rmdiff + rescaleFactor*negmax*rmdiff for i in neg]
    return pos, neg

def rescaleCline(cline, max_var=2):
    rmdiff = random.uniform(1, max_var)
    # move it to start at the origin
    iv = cline[0]
    #print(iv)
    #print(cline[0:3])
    ccline = [i-iv for i in cline]
    #print(ccline[0:3])
    minCline = min(ccline)
    t_cline = [(i+abs(minCline)) for i in ccline]
    maxCline = max(t_cline)
    # more randomness
    n = [random.randint(0, len(t_cline)) for i in range(2)]
    n.sort()
    s1 = t_cline[:n[0]]
    s2 = t_cline[n[0]:n[1]]
    s3 = t_cline[n[1]:]
    rmdiff1 = random.uniform(1, max_var)
    rmdiff2 = random.uniform(1, max_var)
    rmdiff3 = random.uniform(1, max_var)
    b1 = rmdiff1*abs(minCline)/maxCline
    b2 = rmdiff2*abs(minCline)/maxCline
    b3 = rmdiff3*abs(minCline)/maxCline
    ccline = [rmdiff1*float(i)/maxCline - b1 for i in s1] + [rmdiff2*float(i)/maxCline - b2 for i in s2] + [rmdiff3*float(i)/maxCline - b3 for i in s3]
    return ccline

def i_getCline(cline_coeffs, log=False):
    cline_temp = getForm(cline_coeffs)
    cline = rescaleCline(cline_temp)
    return cline

def i_getVines(vine_coeffs, log=False):
    pos = getForm(vine_coeffs)
    neg = getForm(vine_coeffs)
    neg = [-1.0*i for i in neg[::-1]]
    diff = [pos[i] - neg[i] for i in range(len(pos))]
    lbar = 0
    rbar = len(pos)
    for i in range(int(len(pos)*.25)):
    # in the first quarter, cut off anything where neg > pos
        if neg[i] > pos[i]:
            lbar = i
    # in the last quarter, cut off anything where neg > pos
        if neg[-i] > pos[-i]:
            rbar = len(pos) - i
    # slice
    pos = pos[lbar:rbar]
    neg = neg[lbar:rbar]
    pos, neg = rescale(pos, neg, log)
    diff = [pos[i] - neg[i] for i in range(len(pos))]
    mdi = diff.index(max(diff))
    if log:
        print(mdi, max(diff), pos[mdi], neg[mdi])
    # envelope
    if parameters["wsize"] > 0:
        pos = envelope(pos)
        neg = envelope(neg)
    # reslice
    ppos = []
    pneg = []
    if parameters["enforce_minimum_separation"]:
        # enforce a minimum width of the interior
        for i in range(len(pos)):
            # if abs(pos[i] - neg[i]) < parameters["minimum_separation_value"]:
            msep = parameters["minimum_separation_value"]
            # else:
            #     msep = 0
            if pos[i] >= neg[i]:
                ppos.append(pos[i] + msep)
                pneg.append(neg[i] - msep)
            else:
                ppos.append(neg[i] + msep)
                pneg.append(pos[i] - msep)
        diff = [ppos[i] - pneg[i] for i in range(len(ppos))]
    else:
        ppos = pos
        pneg = neg
    if max(diff) > 1.1:
        ppos, pneg = rescale(ppos, pneg)
    return ppos, pneg

def ptsToSVG(inp, loop=False):
    jsvg = Path()
    for i in range(len(inp)-1):
        jsvg.append(Line(start=inp[i], end=inp[i+1]))
    if loop:
        jsvg.append(Line(start=inp[-1], end=inp[0]))
    return jsvg

def SVGWoobler(inp: Path):
    #print()
    n = parameters["glob_n"] # number of sample points. definitely overkill
    cline = getCline()
    # step 1: turn the svg and the cline into N-segment linear approximations
    lin_inp = [inp.point(i/n) for i in range(n)]
    lin_cline = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(cline)), cline)
    #print(lin_inp[0:5])
    #print(lin_cline[0:5])
    # step 2: at each sample point, get the normal vector
    tvs = [inp.unit_tangent(i/n) for i in range(n)]
    nvs = [i.imag-1j*i.real for i in tvs] # im mad
    #print(nvs[0:5])
    # step 3: add the normal vector * cline to the svg
    jit = [lin_inp[i]+lin_cline[i]*nvs[i]*5. for i in range(n)]
    #print(jit[0:5])
    # step 4: convert the new fucked up data back into an svg
    #print(jsvg[0:5])
    return ptsToSVG(jit)

def checkForSI(inp: Path, pn=-1):
    # assuming the input path is piecewise linear
    outpath = Path()
    ints = []
    # need to do this manually, ugh
    i = 0
    while i < len(inp)-2:
        skip = False
        for j in range(i+2, len(inp)-2):
            if inp[i].intersect(inp[j]):
                outpath.append(Line(inp[i].start, inp[j].end))
                i=j
                skip = True
                break
        if not skip:
            outpath.append(inp[i])
        i += 1
        #print(inp[i])
    if debug_opts["log_SI_check_points"]:
        wsvg([inp, outpath], nodes=ints, node_colors='r'*(len(ints)), node_radii=[0.5 for x in ints], colors='kb', stroke_widths=[.5, .25], attributes=None, svg_attributes=None, filename=f'testout/SI-{pn}.svg')
    return SVGtopts(outpath)

def VineSVG(inp: Path, stroke_width: float, pos_scale=5, neg_scale=5, pn=-1):
    n = parameters["glob_n"] # number of sample points
    pos, neg = getVines()
    cline = SVGWoobler(inp)
    # TODO: step 0: pick a random scaling factor
    realwidth = 2. * stroke_width + 1.
    pos_scale = realwidth * random.uniform(pos_scale*parameters["vineVarMin"], pos_scale*parameters["vineVarMax"]) 
    neg_scale = realwidth * random.uniform(neg_scale*parameters["vineVarMin"], neg_scale*parameters["vineVarMax"])
    # step 1 is turn everything into N-segment linear approximations
    lin_inp = [cline.point(i/n) for i in range(n)]
    lin_pos = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(pos)), pos)
    lin_neg = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(neg)), neg)
    # step 2: at each sample point, get the normal vector
    tvs = [inp.unit_tangent(i/n) for i in range(n)]
    nvs = [(i.imag-1j*i.real)*(1+parameters["minimum_separation_value"] if parameters["enforce_minimum_separation"] else 1) for i in tvs] # im mad
    # step 3: add the normal vector * pos to the svg
    pos = [lin_inp[i]+lin_pos[i]*nvs[i]*pos_scale for i in range(n)]
    # step 4: SUBTRACT the normal vector * neg from the svg
    neg = [lin_inp[i]+lin_neg[i]*nvs[i]*neg_scale for i in range(n)]
    if debug_opts["log_pos_neg"]:
        wsvg([inp, ptsToSVG(pos), ptsToSVG(neg)], colors='krb', stroke_widths=[.1, .2, .2], attributes=None, svg_attributes=None, filename=f'testout/RAW-{pn}.svg')
    if debug_opts["log_SI_SVG_check"]:
        checker = [inp, cline, ptsToSVG(pos), ptsToSVG(neg)]
    # step 5 convert these to svgs
    if not debug_opts["bypass_SI_check"]:
        pos, neg = checkForSI(ptsToSVG(pos)), checkForSI(ptsToSVG(neg))
    if debug_opts["log_SI_SVG_check"]:
        checker.append(ptsToSVG(pos))
        checker.append(ptsToSVG(neg))
        wsvg(checker, colors='pgkkrb', stroke_widths=[.25, .25, .5, .5, .1, .1], attributes=None, svg_attributes=None, filename=f'testout/SICHECKED-{pn}-jit.svg')
    # exit()
    # reslice neg to play nice with step 5
    neg = neg[::-1]
    jsvg = Path()
    for i in range(len(pos)-1):
        jsvg.append(Line(start=pos[i], end=pos[i+1]))
    jsvg.append(Line(start=pos[-1], end=neg[0]))
    for i in range(len(neg)-1):
        jsvg.append(Line(start=neg[i], end=neg[i+1]))
    jsvg.append(Line(start=neg[-1], end=pos[0]))
    return jsvg

def processFile(svg_data):
    paths, attributes, _ = svg2paths2(svg_data)
    jpaths = []
    strk_widths = []
    for i in range(len(paths)):
        if(parameters['verbose']):
            print(f"Processing curve {i}")
        #stroke_width = 0
        try:
            style = dict([j.split(":") for j in attributes[i]['style'].split(";")])
            stroke_width = float(style["stroke-width"])
            #print(stroke_width)
        except KeyError:
            print("Invalid SVG data, stroke width not found")
            stroke_width = 0.5
        vsvg = VineSVG(paths[i], stroke_width, pn=svg_data+"-"+str(i))
        jpaths.append(vsvg)
        strk_widths.append(stroke_width)
        if debug_opts["output_svgs"]:
            wsvg(vsvg, colors='k', stroke_widths=[.1], attributes=None, svg_attributes=None, filename=f'{i}-{svg_data}-jit.svg')
    return paths, jpaths, strk_widths, attributes

def i_stampVine(svg: Path, sample_textures):
    # get a random texture, this is all we care about
    ttexture = random.choice(sample_textures)
    # rotate it a random amount
    texture = np.rot90(ttexture, random.randint(0, 3))
    # get the height and width of the vine
    xmin, xmax, ymin, ymax = svg.bbox()
    bboxw = xmax - xmin
    bboxh = ymax - ymin
    bboxmaxdim = max(bboxw, bboxh)
    transform_vector = -1*(xmin + 0.5*bboxw)-1j*(ymin + 0.5*bboxh)
    nPath = svg.translated(transform_vector)
    xmin, xmax, ymin, ymax = nPath.bbox()
    point_version = SVGtopts(nPath)
    texture_size = texture.shape[0]
    rscale_factor = random.randint(int(texture_size*.925), int(texture_size*.975))
    scale = rscale_factor / bboxmaxdim
    spoints = [scale*i for i in point_version]
    nPath = ptsToSVG(spoints, loop=True)
    xmin, xmax, ymin, ymax = nPath.bbox()
    bboxw = xmax - xmin
    bboxh = ymax - ymin
    stamp = np.zeros_like(texture)
    stampsize = stamp.shape
    n2 = int(stampsize[0] / 2)
    test_lines = []
    # massive speedup thanks to awelsh
    intersections = {}
    segs = {}
    for i in range(stampsize[0]):
        if i - n2 + 2 < ymin or i - n2 - 2 > ymax:
            pass
        else:
            opoint = xmin - 1 + 1j*(i - n2)
            epoint = xmax + 1 + 1j*(i - n2)
            lpath = Path(Line(opoint, epoint))
            for (T1, seg1, t1), (T2, seg2, t2) in lpath.intersect(nPath):
                segs[seg1.start.imag] = seg1
                try:
                    intersections[seg1.start.imag].extend([T1])
                except KeyError:
                    intersections[seg1.start.imag] = [T1]
    for k, v in intersections.items():
        if len(v) > 0:
            v.sort()
            assert len(v) % 2 == 0
            pv = [[segs[k].point(v[i]), segs[k].point(v[i+1])] for i in range(0, len(v), 2)]
            for i in pv:
                test_lines.append(Path(Line(i[0], i[1])))
                for j in range(int(i[0].real), int(i[1].real)):
                    stamp[int(k - n2)][j-n2] = texture[int(k - n2)][j-n2]
    out = Image.fromarray(stamp)
    return out.filter(ImageFilter.GaussianBlur(radius=parameters["sigma"]))
    #print(transform_vector)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        prog='ps_harmonics',
        description='Constructs vine data'
    )
    parser.add_argument('files', metavar="file", nargs='+', help='the files to process')
    parser.add_argument('-o', '--outdir', '-d', nargs='?', 
                        default=parameters['output_path'],
                        help=f"output directory for processed files (default {parameters['output_path']})")
    parser.add_argument('--outerLines', nargs='?', 
                        default=data['olines'],
                        help=f"outer line data file (default {data['olines']})")
    parser.add_argument('--centerLines', nargs='?', 
                        default=data['clines'],
                        help=f"center line data file (default {data['clines']})")
    parser.add_argument('--frequencyData', nargs='?', 
                        default=data['freqs'],
                        help=f"frequency data file (default {data['freqs']})")
    parser.add_argument('--imageFiles', nargs='?', 
                        default=data['images'],
                        help=f"image data file (default {data['images']})")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help=f"prints out a lot of info")
    parser.add_argument('-x', '--expert', action='store_true',
                        help=f"allows unsafe actions")
    parser.add_argument('-g', '--svg', action='store_true',
                        help=f"output svg files")
    parser.add_argument('--noImage', action='store_true',
                        help=f"don't output image files")
    parser.add_argument('--DEBUG_SI_SVG_check', action='store_true',
                        help=f"DEBUG ONLY. Output SVG data for self-intersection checking")
    parser.add_argument('-s', '--seed', help="set the random seed used. Needs to be an integer.", default=-1)
    # some day i'll add the rest of this as parameters, but not today
    args = parser.parse_args()
    parameters['output_path'] = args.outdir
    parameters['verbose'] = args.verbose
    debug_opts['expert'] = args.expert

    data['olines'] = args.outerLines
    data['clines'] = args.centerLines
    data['freqs'] = args.frequencyData
    data['images'] = args.imageFiles

    debug_opts["output_pngs"] = not args.noImage
    debug_opts["output_svgs"] = args.svg

    debug_opts["log_SI_SVG_check"] = args.DEBUG_SI_SVG_check

    try:
        assert debug_opts["bypass_SI_check"] == False
    except AssertionError:
        if not debug_opts["expert"]:
            print("ERROR: You are bypassing the self-intersection checking code. If you're certain, rerun with the -x flag.")
            traceback.print_exc()
            exit()
        else:
            print("WARNING: You are bypassing the self-intersection checking code.")
    try:
        assert debug_opts["output_pngs"] == True or debug_opts["output_svgs"] == True
    except AssertionError:
        print("ERROR: You need to output something.")
        traceback.print_exc()
        exit()
    makedirs(parameters['output_path'], exist_ok=True)
    try: 
        sd = int(args.seed)
    except ValueError:
        print(f"{args.seed} is not a valid seed. Using a random seed.")
        sd = -1
    setSeed(sd)
    vine_coeffs, cline_coeffs, textures = getSetup()
    def getCline(): return i_getCline(cline_coeffs)
    def getVines(): return i_getVines(vine_coeffs)
    def stampVine(svg: Path): return i_stampVine(svg, textures)

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
        paths, jpaths, strk_widths, _ = processFile(i)

        for j in range(len(jpaths)):
            if(parameters['verbose']):
                print(f"Outputting curve {j}")
            if debug_opts["output_pngs"]:
                ret = stampVine(jpaths[j])
                ret.save(f"{parameters['output_path']}{basename(i)[0]}-{j}-jit.png")
            if debug_opts["output_svgs"]:
                wsvg(jpaths, colors='k'*len(jpaths), stroke_widths=[.5 for k in range(len(jpaths))], attributes=None, svg_attributes=None, filename=f'{parameters["output_path"]}{i}-jit.svg')
