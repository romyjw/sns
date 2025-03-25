# SphericalNS Setup Instructions

## Prerequisites
Ensure you have the necessary dependencies installed, including:
- `cmake`
- `make`
- `Python` (version compatible with `0automatic.prepare_overfit_Balin.py`)

## Installation Steps

### 1. Create the `SphericalNS` Folder
```sh
mkdir SphericalNS
cd SphericalNS
```

### 2. Clone Required Repositories
#### Clone the modified `surface-maps-via-adaptive-triangulations` repository:
```sh
git clone https://github.com/romyjw/ROMYsmvat
```
#### Clone the `SphericalFour` repository and rename it:
```sh
git clone https://github.com/romyjw/SphericalFour spherical2
```

### 3. Compile the Code in `ROMYsmvat`
```sh
cd ROMYsmvat
mkdir build
cd build
cmake ..
make -j4
```

### 4. Prepare the Data Folder
Create a `data` folder inside `spherical2`:
```sh
mkdir -p SphericalNS/spherical2/data
```
Place a genus 0 `.obj` mesh in the `data` folder, e.g., `MAX10606.obj`.

### 5. Run the Preparation Script
Navigate to the neural surfaces directory:
```sh
cd SphericalNS/spherical2/neural-surfaces-main
```
Run the script:
```sh
python -m 0automatic.prepare_overfit_Balin.py
```
When prompted for the mesh name, provide the name **without** the `.obj` extension, e.g., `MAX10606`.

### 6. Verify the Preparation
If the script runs successfully, it should:
- Normalize the mesh
- Generate the necessary `.json` file
- Output the command to run the overfitting training

You're now ready to proceed with training and further experiments! 🚀

---

## README File
To create a `README.md` file for GitHub, save the following content as `README.md` in your repository:

```markdown
# SphericalNS Setup Instructions

## Prerequisites
Ensure you have the necessary dependencies installed, including:
- `cmake`
- `make`
- `Python` (version compatible with `0automatic.prepare_overfit_Balin.py`)

## Installation Steps

### 1. Create the `SphericalNS` Folder
```sh
mkdir SphericalNS
cd SphericalNS
```

### 2. Clone Required Repositories
#### Clone the modified `surface-maps-via-adaptive-triangulations` repository:
```sh
git clone https://github.com/romyjw/ROMYsmvat
```
#### Clone the `SphericalFour` repository and rename it:
```sh
git clone https://github.com/romyjw/SphericalFour spherical2
```

### 3. Compile the Code in `ROMYsmvat`
```sh
cd ROMYsmvat
mkdir build
cd build
cmake ..
make -j4
```

### 4. Prepare the Data Folder
Create a `data` folder inside `spherical2`:
```sh
mkdir -p SphericalNS/spherical2/data
```
Place a genus 0 `.obj` mesh in the `data` folder, e.g., `MAX10606.obj`.

### 5. Run the Preparation Script
Navigate to the neural surfaces directory:
```sh
cd SphericalNS/spherical2/neural-surfaces-main
```
Run the script:
```sh
python -m 0automatic.prepare_overfit_Balin.py
```
When prompted for the mesh name, provide the name **without** the `.obj` extension, e.g., `MAX10606`.

### 6. Verify the Preparation
If the script runs successfully, it should:
- Normalize the mesh
- Generate the necessary `.json` file
- Output the command to run the overfitting training

You're now ready to proceed with training and further experiments! 🚀
```

