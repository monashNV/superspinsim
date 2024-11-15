.\activate_venv.ps1
nsys profile --trace cuda --gpu-metrics-devices=all --cuda-memory-usage true --force-overwrite true --output profile/profile python .
