# ps_vine
# tools for constructing razorvine images in the style of the AD&D planescape books

# frequencies to use for analysis
# MUST be a power of 2
# should be >= 2**10
fourier_n = 2**10
magic_frequencies = [x/fourier_n for x in range(0, int(fourier_n/2))]+[x/fourier_n for x in range(-1*int(fourier_n/2), 0)]
def getFreqs():
    return magic_frequencies

def parametrizeSection(path, limit, offset=[0,0]):
    div = 1./limit
    out = []
    for i in range(limit):
        tmp = path.point(i*div)
        out.append([tmp.real+offset[0], tmp.imag+offset[1]])
    return out
