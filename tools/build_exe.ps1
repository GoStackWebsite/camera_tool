# Builds the camera engine into a portable folder: dist\CameraEngine\CameraEngine.exe
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\build_exe.ps1
# Congruence Eye is not part of it. .env and data\ of this PC (camera passwords, API key, footfall) are never copied.
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root '.venv\Scripts\python.exe'
$out = Join-Path $root 'dist\CameraEngine'

if (-not (Test-Path $py)) { throw 'Virtual environment missing. Run setup.bat first.' }
# a rebuild replaces the whole folder: never over one that has been in use
if (Test-Path (Join-Path $out 'data')) { throw "$out holds a data folder (it has been run). Move it elsewhere first." }

& $py -m pip install --quiet --disable-pip-version-check pyinstaller
if ($LASTEXITCODE) { throw 'pip install pyinstaller failed' }
& $py -m PyInstaller (Join-Path $PSScriptRoot 'camera_engine.spec') --noconfirm --clean --log-level WARN `
    --distpath (Join-Path $root 'dist') --workpath (Join-Path $root 'build')
if ($LASTEXITCODE) { throw 'PyInstaller failed' }

# the layout run.bat sees: the YOLO weights in the root, the store models in models\
foreach ($file in 'yolo11m.pt', 'yolo11n-cls.pt') { Copy-Item (Join-Path $root $file) $out }
New-Item -ItemType Directory -Force (Join-Path $out 'models') | Out-Null
foreach ($file in 'staff_cls.pt', 'brand_cls.pt', 'face_detection_yunet_2023mar.onnx', 'genderage.onnx') {
    Copy-Item (Join-Path $root "models\$file") (Join-Path $out 'models')
}

# settings for a new PC: no camera of this site, cameras are added in the dashboard
$lines = (Get-Content (Join-Path $root '.env.example')) -replace '^RTSP_URL=.*', 'RTSP_URL='
$lines += '', '# Packaged exe', 'OPEN_BROWSER=1            # 0 = do not open the dashboard at start'
Set-Content (Join-Path $out '.env') $lines -Encoding ascii

$size = (Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
'Built {0}\CameraEngine.exe ({1:N1} GB)' -f $out, $size
