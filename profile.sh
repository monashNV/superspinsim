nsys profile --trace cuda --gpu-metrics-devices=all --cuda-memory-usage true --force-overwrite true --output profile/profile python .
nsys export --type=hdf --force-overwrite=True --output=profile/profile.h5 profile/profile.nsys-rep
