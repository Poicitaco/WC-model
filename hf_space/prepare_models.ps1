$ErrorActionPreference = "Stop"
$source = Join-Path $PSScriptRoot "..\models"
$target = Join-Path $PSScriptRoot "models"

New-Item -ItemType Directory -Force -Path $target | Out-Null
Copy-Item (Join-Path $source "random_forest_baseline.pkl") $target -Force
Copy-Item (Join-Path $source "random_forest_tuned.pkl") $target -Force
Copy-Item (Join-Path $source "xgboost_baseline.pkl") $target -Force
Copy-Item (Join-Path $source "xgboost_tuned.pkl") $target -Force

Write-Output "Copied all four models into hf_space/models."
