# Spherical Neural Surfaces
![Frogs coloured by differential quantities](teaser.png?raw=true "SNS Frogs")
This is the official implentation of Neural Geometry Processing via Spherical Neural Surfaces (Eurographics 2025).
Please see the project webpage: https://geometry.cs.ucl.ac.uk/projects/2025/sns/.

This repository also contains code to generate some smooth genus-0 analytic test shapes that we designed for the evaluation of this project - more details below. We welcome you to use any of these shapes in your own project, but please give credit.

## Installation Steps

#### 1. Create the `SphericalNS` Folder
```sh
mkdir SphericalNS
cd SphericalNS
```

#### 2. Clone the `sns` repository:
```sh
git clone https://github.com/romyjw/sns
```
#### 3. Make your conda environment.

Necessary packages are listed in ```environment.yml```.
Run ```conda env create -f environment.yml ``` or install packages manually one-by-one.
(This is the environment for visualisation, on CPU.)
For training, we recommend that you use CUDA. 
If you have issues installing pyrender, try [these instructions](https://pyrender.readthedocs.io/en/latest/install/) or [these instructions](https://github.com/smartgeometry-ucl/COMP0119_24-25/tree/main/lab_demos/Tutorial%201%20-%20%20coding_framework#install-pyrender).

The visualisation scripts have been tested locally on Mac.
The training scripts have been tested on an Ubuntu machine, running on GPU with CUDA.

## Visualising Differential Quantities on an SNS

The simplest way to visualise an SNS is to push a sphere-mesh through the SNS map. Then we can see the shape of the SNS, and we can display curvatures etc. as vertex-colours.

#### 1. Generate some sphere-meshes.
Run the sphere-generation script - ```python make-sphere-mesh.py``` - to generate some sphere meshes. The denser, the better. By default, the script will try to generate icosphere meshes up to 9 subdivisions of an icosahedron, for high resolution visualisation. Spheres get stored in the folder ```data/analytic/sphere/ ```.

#### 2. Visualise
To visualise an SNS, you will need:
- the model weights for the MLP, e.g. ```data/SNS/MAX10606/ ```
- a sphere mesh at the resolution that you want to display, e.g. ```data/analytic/sphere/sphere6.obj ```

Check that you have these, check that the filepaths are correct in ```visuals/visOverfit.py ```, then run e.g. ```sh python -m visuals.visOverfit  MAX10606 6 ```. The number refers to which sphere mesh to use; ```sphere6.obj``` is the level 6 icosphere.

#### 3. Crossfields
By changing settings within ```neural_surfaces-main/visuals/visOverfit.py ``` you can generate obj files for maximum and minimum curvature directions. These will be stored in, e.g. ```data/visualisation/MAX10606 ```
Open ```crossfield.obj``` and ```icosphere_MAX10606.obj``` together (we recommend [https://www.meshlab.net/](MeshLab) ), to see the coloured crossfield overlaid on the SNS shape.

## Overfit your own SNS


### 1. Prepare the Data Folder

Place a genus-0 `.obj` mesh in the `data` folder, e.g., `MAX10606.obj`.

### 2. Spherical Mesh Embedding
Generate a spherical mesh embedding of the mesh. In the SNS paper, we use Multi-Resolution sphere embedding - code is available from Schmidt et. al. here: https://github.com/patr-schm/surface-maps-via-adaptive-triangulations .
Place a spherical mesh-embedding of the shape in the data folder, and name it e.g. MAX10606_final_embedding.obj


### 3. Run the Preparation Script
Navigate to the neural surfaces directory:
```sh
cd SphericalNS/sns/neural-surfaces-main
```
Edit the filepath in this script: 0automatic/prepare_overfit.py .

Run the script:
```sh
python -m 0automatic.prepare_overfit.py
```
When prompted for the mesh name, provide the name **without** the `.obj` extension, e.g., `MAX10606`.

### 4. Verify the Preparation
If the script runs successfully, it should:
- Normalize the mesh by bounding box size (which is saved as ```data/MAX10606_nB.obj```, for example )
- Generate a `.json` file in ```neural_surfaces-main/experiment_configs/overfit/``` (which you can customise if you choose)
- Output the exact command needed to run the overfitting

### 5. Train

Run the command outputted by the preparation script. E.g. 
```python -m mains.training experiment_configs/overfit/MAX10606.json``` .



## Analytic Test Shapes


Text files containing the formulae for several genus-0 analytic test shapes are provided in the ```data/analytic``` directory. After you have created some sphere meshes (as explained above, with ```make-sphere-mesh.py```) then you can run, e.g. ```python analytic_shape SMALLFLOWER 7``` to generate a mesh of the SMALLFLOWER surface, as a deformation of the level 7 icosphere.
![Analytic Genus-0 Shapes](shapes.png?raw=true "Analytic Genus-0 Shapes")

---

