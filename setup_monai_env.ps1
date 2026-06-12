param(
    [string]$EnvName = "monai",
    [string]$PythonVersion = "3.11",
    [ValidateSet("gpu", "cpu")]
    [string]$Mode = "gpu",
    [ValidateSet("cu128", "cu126", "cu118")]
    [string]$CudaWheel = "cu128"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-Command {
    param([string]$Name)
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Step "Checking conda"
if (-not (Test-Command "conda")) {
    throw "conda was not found. Please install Anaconda or Miniconda, then reopen PowerShell."
}

Write-Step "Conda version"
conda --version

$envList = conda env list
$envExists = $envList | Select-String -Pattern "^\s*$([regex]::Escape($EnvName))\s"

if (-not $envExists) {
    Write-Step "Creating conda environment: $EnvName, Python $PythonVersion"
    conda create -n $EnvName python=$PythonVersion -y
} else {
    Write-Step "Environment '$EnvName' already exists; reusing it"
}

Write-Step "Installing packages into environment: $EnvName"

$pipUpgrade = "python -m pip install --upgrade pip setuptools wheel"
conda run -n $EnvName powershell -NoProfile -ExecutionPolicy Bypass -Command $pipUpgrade

if ($Mode -eq "gpu") {
    $torchIndexUrl = "https://download.pytorch.org/whl/$CudaWheel"
    Write-Step "Installing PyTorch GPU build from $torchIndexUrl"
    $torchInstall = "python -m pip install torch torchvision torchaudio --index-url $torchIndexUrl"
} else {
    Write-Step "Installing PyTorch CPU build"
    $torchInstall = "python -m pip install torch torchvision torchaudio"
}

conda run -n $EnvName powershell -NoProfile -ExecutionPolicy Bypass -Command $torchInstall

Write-Step "Installing MONAI with common medical imaging dependencies"
$monaiInstall = 'python -m pip install "monai[nibabel,skimage,pillow,tqdm,matplotlib,tensorboard,pydicom,scipy,pandas,einops]"'
conda run -n $EnvName powershell -NoProfile -ExecutionPolicy Bypass -Command $monaiInstall

Write-Step "Verifying installation"
$verifyScript = @'
import torch
import monai

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("monai:", monai.__version__)
print("")
monai.config.print_config()
'@

$encodedVerifyScript = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($verifyScript))
conda run -n $EnvName python -c "import base64; exec(base64.b64decode('$encodedVerifyScript').decode('utf-8'))"

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Activate the environment with:"
Write-Host "  conda activate $EnvName" -ForegroundColor Yellow
