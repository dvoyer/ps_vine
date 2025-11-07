# ps_vine
Deconstruct samples of “vines” from AD&D Planescape books (or other reference textures/illustrations), then reconstruct new vines based on the same levels of “roughness”.

This project takes traced SVGs of vine-like patterns, extracts geometric and frequency data from their outlines, and then generates new vines that follow arbitrary SVG center lines.

## Installation

- Python 3.9+
- ``pip install svgpathtools numpy scipy Pillow``

    
## Project Structure

```bash
ps_vine.py             # Core math utilities: frequency bins, path sampling
ps_deconstruct.py      # Step 1: extract frequency data from SVG outlines
ps_prep_data.py        # Step 2: consolidate features and textures
ps_harmonics.py        # Step 3: generate and render new vines
textures/              # Place square (even-sized) reference images here
raw_data/              # Step 1 output
data/                  # Step 2 output (used by ps_harmonics)
output/                # Step 3 output
```
## Workflow
### Stage 0: Deconstruction
You only need to do this if you want to add your own frequency data from your own vines. Otherwise, this isn't necessary.

1. Start with a reference image.
2. Open it in Inkscape (or any vector tool) and trace its outline. Inkscape has automatic tools for this.
3. Separate it into two paths:
- The outer contour of the vine
- The centerline running through it
- TODO: automate this?
Save as path.svg.

4. Run ``python ps_deconstruct.py -o raw_data/ path.svg``
- Output is at ``raw_data/FreqData-path.pkl``

### Stage 0.5: Compilation
You only need to run this step initially, or when you add more frequency data or update the included images for stamping.
1. Run: ``python ps_prep_data.py -d raw_data/ -o data/ -i textures/``.
If you don’t need image data (SVG-only output later), add ``--noimg``

### Step 1: Generation
1. Create an svg file with a centerline you want the vine to follow.
2. Run: ``python ps_harmonics.py path/to/centerline.svg``.
- If you want svg files too, add ``-g`` or ``--svg``.
- If you don't want any raster images, add ``--noImage``.
3. Outputs PNGs in ``output/`` by default.


## Parameters
``ps_harmonics.py`` has the following parameters embedded in it. These are currently unsettable by the CLI. They're set to defaults that look passable. Tweak these to make vines thicker, more erratic, or smoother.

|            Parameter            |    Default   |                             Description                             |
|:-------------------------------:|:------------:|:-------------------------------------------------------------------:|
|            ``wsize``            |      32      |            Window size for envelope, smooths outer paths            |
|      ``middle_freq_scale``      |     0.15     | Reduce mid-band frequencies to this amount to reduce mid-band noise |
|        ``amp_filter[0]``        |  [0.33, 0.5] |               Gaussian filter on Fourier coefficients               |
|        ``amp_filter[1]``        | [0.25, 0.75] |           Another Gaussian filter on Fourier coefficients           |
|            ``sigma``            |     0.75     |               Gaussian blur radius for raster outputs               |
| ``vineVarMin`` / ``vineVarMax`` |  0.85 / 1.1  |                    Random vine thickness scaling                    |
|   ``minimum_separation_value``  |     0.05     |          The minimum permitted internal dimension of a vine         |

TODO: make these settable via expert mode

## CLI Options
### All Files
|         Option         |            Default            |        Description        |
|:----------------------:|:-----------------------------:|:-------------------------:|
|  ``-o`` / ``--outdir`` | ``parameters['output_path']`` |      The output path      |
| ``-v`` / ``--verbose`` |             False             | Prints out extensive logs |

### ps_prep_data.py
|        Option        |            Default           |                  Description                 |
|:--------------------:|:----------------------------:|:--------------------------------------------:|
| ``-d`` / ``--indir`` |  ``parameters['data_path']`` |         Input directory for raw files        |
|  ``-i`` / ``--img``  | ``parameters['image_path']`` | Input directory for raster images for stamps |
| ``-n`` / ``--noImg`` |             False            |            Does not process images           |

### ps_harmonics.py
|         Option        |       Default      |                 Description                 |
|:---------------------:|:------------------:|:-------------------------------------------:|
|    ``--outerLines``   | ``data['olines']`` |             Outer line data file            |
|   ``--centerLines``   | ``data['clines']`` |            Center line data file            |
|  ``--frequencyData``  |  ``data['freqs']`` |             Frequency data file             |
|    ``--imageFiles``   | ``data['images']`` |               Image data file               |
|   ``-g`` / ``--svg``  |        False       |               Output SVG files              |
|     ``--noImage``     |        False       |              Don't output a png             |
|  ``-s`` / ``--seed``  |         -1         | Positive integer, sets the random seed used |
| ``-x`` / ``--expert`` |        False       |            Allows unsafe actions            |

At this time, expert mode only gatekeeps the self-intersection code bypass, which must be manually changed in the debug options.
## License

[GPLv3](https://choosealicense.com/licenses/gpl-3.0)

## Acknowledgments
Built with [svgpathtools](https://pypi.org/project/svgpathtools/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Pillow](https://pypi.org/project/pillow/).

Thanks to TSR for the idea and the setting.