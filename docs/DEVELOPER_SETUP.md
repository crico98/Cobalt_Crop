# Developer setup

## Supported baseline

The current development workstation baseline is:

| Component | Current baseline |
|---|---|
| Operating system | Windows 11 |
| Python | 3.12.0 |
| Flask | 3.1.3 |
| Ultralytics | 8.4.203 |
| OpenCV | 5.0.0 |
| NumPy | 2.5.4 |
| Pillow | 13.0.0 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| GPU | NVIDIA RTX 3070 Ti, 8 GB |
| CUDA availability | True in the current workstation environment |

The versions above describe the current workstation baseline. requirements.txt is the project dependency declaration; PyTorch and torchvision are installed separately so a developer can select the correct CPU or CUDA wheel for their machine.

## 1. Clone the current development branch

The repository is private and currently defaults to main. The active crop-pipeline implementation is on feature/view-specific-crop-pipeline.

A collaborator must first be invited to the private repository and accept the invitation.

~~~powershell
git clone --branch feature/view-specific-crop-pipeline https://github.com/crico98/Standalone-Crop.git
cd Standalone-Crop
~~~

## 2. Create and activate a virtual environment

Install Python 3.12, then run from the repository root:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
~~~

The version check should report Python 3.12.x.

## 3. Install PyTorch for the local machine

PyTorch is deliberately kept out of requirements.txt because CPU and CUDA installations use different wheels.

For an NVIDIA setup compatible with CUDA 12.8, the current workstation used:

~~~powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
~~~

For CPU-only machines or a different CUDA environment, use the official PyTorch installation selector and choose the matching Python, operating system, and compute platform. Do not install a CPU-only build over an existing working CUDA installation.

Verify the installation:

~~~powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

## 4. Install project dependencies

~~~powershell
python -m pip install -r requirements.txt
~~~

## 5. Supply model checkpoints

The Face V2 checkpoint is present in the repository at:

~~~text
models/face_combined_v2_finetune/weights/best.pt
~~~

Top and Isometric checkpoints are not stored in Git. Obtain approved copies and place them at:

~~~text
models/top_view/weights/best.pt
models/isometric_view/weights/best.pt
~~~

The app reports whether each checkpoint is available. A missing Top or Isometric checkpoint prevents inference for that view; it does not fall back to another model.

Do not commit model weights, training datasets, proprietary source photos, or annotation manifests unless the repository owner has explicitly approved their inclusion.

### Optional Top / Isometric annotation editor

The browser annotation editor at /annotate is separate from the crop upload workflow. It expects local source images under dataset/train/images and the local annotation manifest at dataset/top_isometric_annotations.json. These proprietary inputs are not included in the repository. The editor can be used only after the owner has supplied the approved local dataset and manifest. The editor and training utilities are not required to run inference on an uploaded photo batch.

## 6. Start the application

~~~powershell
python app.py
~~~

Open http://127.0.0.1:5000.

Optional environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| CROPPER_MAX_UPLOAD_MB | 50000 | Maximum combined upload size in MB |
| CROPPER_MAX_FILES | 500 | Maximum number of photos in one batch |

## 7. Confirm the environment

~~~powershell
python -c "import flask, ultralytics, cv2, numpy, PIL; print('Flask', flask.__version__); print('Ultralytics', ultralytics.__version__); print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__); print('Pillow', PIL.__version__)"
python -m unittest discover -s tests -v
~~~

See docs/TESTING.md for what the automated suite does and does not validate.
