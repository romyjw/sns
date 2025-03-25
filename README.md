# SphericalNS Setup Instructions

## Environment
You can see the required python packages, in the environment.yml file. 
It is recommended to run with CUDA.

## Installation Steps

### 1. Create the `SphericalNS` Folder
```sh
mkdir SphericalNS
cd SphericalNS
```


#### Clone the `sns` repository:
```sh
git clone https://github.com/romyjw/sns
```

### 2. Prepare the Data Folder
Create a `data` folder inside `sns`:
```sh
mkdir data
```
Place a genus-0 `.obj` mesh in the `data` folder, e.g., `MAX10606.obj`.

### 3. Spherical Mesh Embedding
Generate a spherical mesh embedding of the mesh. In the SNS paper, we use Multi-Resolution sphere embedding - code is available from Schmidt et. al. here: https://github.com/patr-schm/surface-maps-via-adaptive-triangulations .
Place a spherical mesh-embedding of the shape in the data folder, and name it e.g. MAX10606_final_embedding.obj



### 4. Run the Preparation Script
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

### 6. Verify the Preparation
If the script runs successfully, it should:
- Normalize the mesh
- Generate the necessary `.json` file
- Output the exact command needed to run the overfitting training

### 7. Train

Just run the command outputted by the preparation script. E.g. 
python -m mains.training experiment_configs/overfit/MAX10606.json .

---

