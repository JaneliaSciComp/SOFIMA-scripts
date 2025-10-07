wrapper scripts for SOFIMA to batch jobs to the janelia cluster.

the entry point is 2-planes-align.sh, which submits two jobs, the first
(2-planes-flow-mesh.py) handles the GPU intensive portions of the algorithm, and a
second dependent one (2-planes-invmap.py) which uses only the CPU and also a lot
of RAM.  data loading and saving code is in a plugin (see data-\*.py).

a similar workflow exists for two volumes, which aligns the overlapping portions
instead of just two adjacent slices.

multiple planes each with multiple tiles can be stitched and aligned with
1-plane-\* and N-planes-\*, respectively.

# installation

You need a working conda installation, e.g., through
[miniforge](https://github.com/conda-forge/miniforge).
Then, from the root of this repository, run:

```
conda env create -f environment.yaml
conda activate multi-sem
```

# basic use

manually edit 2-planes-align.sh to set `basepath` to the location of this
repository, and the arguments to `bsub` to the number of slots required, etc.

manually edit one of the data-\*.py (e.g. to set `zbase` to the location of
your data and `itop` and `ibot` to the two slices you want to align).
alternatively, write your own data-loader plugin defining the same functions
that are in data-\*.py.

then execute: `./2-planes-align.sh <data-loader> <patch-size> <stride> <batch-size>`

\<patch-size\> and \<stride\> are in units of pixels and set the XY spatial
context used for flow field estimation and the XY distance between centers of
adjacent patches, respectively.

set <batch-size> such that all GPU cores are used without exceeding GPU RAM

# limitations

designed to only align the bottom of one 3D block to the top of another

uses the zarr folder of the input data as scratch space for intermediate results 

flow fields are only calculated at a single resolution

# development

Unit tests are in 2-{planes,volumes}-test.py and use data-test-2-{planes,volumes}.py.

### Aligning two planes

Run
```bash
python 2-planes-flow-mesh.py data-test-2-planes 16 8 1
python 2-planes-invmap.py data-test-2-planes 16 8
python 2-planes-test.py data-test-2-planes 16 8
```
  
The following figure should be generated (overlay.png):

![output of unit tests](overlay.png)

### Aligning a full volume

Run
```bash
python 2-volumes-flow-mesh.py data-test-2-volumes 32 8 2
python 2-volumes-invmap.py data-test-2-volumes 32 8
python 2-volumes-test.py data-test-2-volumes 32 8
```

There should be two figures generated (overlay-xy.png and overlay-xz.png) that
show the alignment in XY and XZ planes, respectively.

### stitching and aligning the aphid salivary gland data set

to stitch tiles within a plane, edit "data-aphid-1-plane.py" to specify how
many pixels to trim from each edge of the tiles in `crop`.  and note that a 2x3
arrangement of tiles within a plane is currently assumed.  then run

```bash
./1-plane-stitch.sh <level> <patch_size> <stride> <k0> <k> <min_z> <max_z> <outpath>
```

reasonable defaults for `level` etc. are 0 50 5 0.01 0.1 10770 10780.

wait for the cluster jobs to finish (a few minutes).

then, to align planes with each other, note that `url` for the aphid dataset is
hard-coded in "data-aphid-N-planes.py".  then on your workstation run

```bash
./N-planes-align.sh "data-aphid-N-planes" <basepath> <min-z> <max-z> <patch-size> <stride> <scales> <k0> <k> <repeat> <batch-size> <chunk-size>
```

where reasonable defaults for `patch-size` etc. are 50 5 1,2 0.01 0.1 1 2048 128.

internally, 1-plane-stitch.sh calls 1-plane-stitch.py and the data loading code
is architected as a plugin in data-aphid-1-plane.py.

similarly, N-planes-align.sh calls N-planes-{flow,mesh,invmap,warp}.py
successively with the aphid-data-N-planes.py data loading plugin.  each python
script saves intermediates results to an .npy file.  the final output is a
zarr.
