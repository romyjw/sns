# SphericalNS Setup Instructions

## Environment
You can see the required python packages, in the environment.yml file. 
It is recommended to run with CUDA.

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

Necessary packages are listed in environment.yml
For training, we recommend that you use cuda.

## Visualising Differential Quantities on an SNS

The simplest way to visualise an SNS is to push a sphere-mesh through the SNS map. Then we can see the shape of the SNS, and we can display curvatures etc. as vertex-colours.

#### 1. Generate some sphere-meshes.
Run the script make-sphere-mesh.py to generate some sphere meshes. The denser, the better. By default, the script will try to generate icosphere meshes up to 9 subdivisions of an icosahedron, for high resolution visualisation.

#### 2. Visualise
To visualise an SNS, you will need:
- the model weights for the MLP, e.g. ```sh data/SNS/MAX10606/ ```
- a sphere mesh at the resolution that you want to display, e.g. ```sh data/analytic/sphere/sphere6.obj ```

Check that you have these, check that the filepaths are correct in ```sh visuals/visOverfit.py ```, then run e.g. ```sh python -m visuals.visOverfit  MAX10606 6 ```. The number refers to which sphere mesh to use; sphere6.obj is the level 6 icosphere.

#### 3. Crossfields
By changing settings within ```sh neural_surfaces-main/visuals/visOverfit.py ``` you can generate obj files for maximum and minimum curvature directions. These will be stored in, e.g. ```sh data/visualisation/MAX10606 ```
Open ```sh crossfield.obj``` and ```icosphere_MAX10606.obj``` together (we recommend https://www.meshlab.net/ ), to see the coloured crossfield overlaid on the SNS shape.

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
- Normalize the mesh
- Generate the necessary `.json` file
- Output the exact command needed to run the overfitting training

### 5. Train

Just run the command outputted by the preparation script. E.g. 
python -m mains.training experiment_configs/overfit/MAX10606.json .

---

