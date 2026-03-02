param(
    [string]$EnvName = "butterfly-classification",
    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

conda create -n $EnvName python=$PythonVersion -y

conda install -n $EnvName -y -c conda-forge numpy pandas scikit-learn matplotlib seaborn

conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install tensorflow

conda run -n $EnvName python -m pip install -r requirements.txt

Write-Host "activate conda with : conda activate $EnvName"