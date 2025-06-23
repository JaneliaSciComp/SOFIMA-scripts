wrapper scripts for SOFIMA to batch jobs to the janelia cluster.

the entry point is em-alignment.sh, which submits two jobs, the first
(em-alignment1.py) handles the GPU intensive portions of the algorithm, and a
second dependent one (em-alignment2.py) which uses only the CPU and also a lot
of RAM.

# installation

```
conda create -n multi-sem
conda install jax tensorstore
conda install matplotlib  # optional
pip install git+https://github.com/google-research/sofima
```

sofima also installs numpy and connectomics which these scripts directly use

matplotlib is needed only for debugging purposes.  uncomment the lines of
code which do the plotting if problems arise.

# basic use

manually edit em-alignment.sh to set `basepath` to the location of this
repository, and the arguments to `bsub` to the number of slots required, etc.

manually edit `em-alignment{1,2}.py` to set `zbase` to the location of your
data and the last (3rd) indices into `ttop` and `tbot` to the two slices you
want to align.

then execute: `./em-alignment.sh <patch-size> <stride> <batch-size>`

\<patch-size\> and \<stride\> are in units of pixels and set the XY spatial
context used for flow field estimation and the XY distance between centers of
adjacent patches, respectively.

set <batch-size> such that all GPU cores are used without exceeding GPU RAM

# limitations

designed to only align the bottom of one 3D block to the top of another

uses the zarr folder of the input data as scratch space for intermediate results 

path to data is hard-coded so difficult to apply to different data

flow fields are only calculated at a single resolution
