nsys profile --trace cuda --cuda-memory-usage true --force-overwrite true --output profile/profile python .
nsys export --type=hdf --force-overwrite=True --output=profile/profile.h5 profile/profile.nsys-rep
python read_profile.py
