python -m data_analysis.hdf5_to_zarr_converter_parrel -e "PullCube" --resize
sleep 10
python -m data_analysis.hdf5_to_zarr_converter_parrel -e "LiftPegUpright" --resize
sleep 10
python -m data_analysis.hdf5_to_zarr_converter_parrel -e "PegInsertionSide" --resize
sleep 10
python -m data_analysis.hdf5_to_zarr_converter_parrel -e "PlaceSphere" --resize