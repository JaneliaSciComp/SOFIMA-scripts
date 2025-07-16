wrapper scripts for SOFIMA to batch jobs to the janelia cluster.

the entry point is em-alignment.sh, which submits two jobs, the first
(em-alignment1.py) handles the GPU intensive portions of the algorithm, and a
second dependent one (em-alignment2.py) which uses only the CPU and also a lot
of RAM.  data loading and saving code is in a plugin, which defaults to data.py.

# installation

```
conda create -n multi-sem
conda install jax tensorstore
conda install matplotlib  # optional
pip install git+https://github.com/google-research/sofima
```

sofima also installs numpy and connectomics which these scripts directly use

matplotlib is needed only for testing purposes.  see em-alignment3.py and
development section below.

# basic use

manually edit em-alignment.sh to set `basepath` to the location of this
repository, and the arguments to `bsub` to the number of slots required, etc.

manually edit `data.py` to set `zbase` to the location of your data and `itop`
and `ibot` to the two slices you want to align.  alternatively, write your
own data-loader plugin defining the same functions that are in data.py.

then execute: `./em-alignment.sh <data-loader> <patch-size> <stride> <batch-size>`

\<patch-size\> and \<stride\> are in units of pixels and set the XY spatial
context used for flow field estimation and the XY distance between centers of
adjacent patches, respectively.

set <batch-size> such that all GPU cores are used without exceeding GPU RAM

# limitations

designed to only align the bottom of one 3D block to the top of another

uses the zarr folder of the input data as scratch space for intermediate results 

flow fields are only calculated at a single resolution

# development

unit tests are in em-alignment3.py and use data-test.py.  the following figure
should be generated:

![output of unit tests](overlay.png)
