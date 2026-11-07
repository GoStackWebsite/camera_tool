# PyInstaller recipe for CameraEngine.exe. Run it through tools\build_exe.ps1, which also adds the model files.
# A folder build on purpose: as one file, the exe would unpack several GB of CUDA libraries to %TEMP% at every start.
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

ROOT = Path(SPECPATH).parent

# ultralytics asks for these versions at import and when tracking starts
METADATA = ("lap", "numpy", "opencv-python", "pillow", "torch", "torchvision", "ultralytics")

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    # torchvision loads these by path (YOLO's NMS lives in _C_stable) and the stock hook still looks for the old name _C
    binaries=collect_dynamic_libs("torchvision", search_patterns=["*.dll", "*_stable.pyd"]),
    datas=[
        (str(ROOT / "app" / "static"), "app/static"),
        (str(ROOT / "app" / "tracker.yaml"), "app"),
        *(item for name in METADATA for item in copy_metadata(name)),
    ],
    hiddenimports=["app.server"],    # uvicorn imports it by name, the analysis cannot see that
    # training and export tools: ultralytics and torch only import them on paths the engine never takes
    excludes=["polars", "_polars_runtime_32", "tkinter", "IPython", "pytest", "tensorboard", "onnx"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CameraEngine", console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="CameraEngine", upx=False)
