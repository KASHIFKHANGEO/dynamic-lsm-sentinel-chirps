# Dynamic Landslide Susceptibility Mapping: A Spatiotemporal Machine Learning Framework Using Sentinel-2 and CHIRPS with Temporal Holdout Validation

This repository contains the code for the manuscript:
"Dynamic Landslide Susceptibility Mapping: A Spatiotemporal Machine Learning Framework Using Sentinel-2 and CHIRPS with Temporal Holdout Validation"

 Repository Structure
dynamic-lsm-sentinel-chirps/888888888888888888888888888
README.md requirements.txt

gee/
        export_temporal.js # GEE script for temporal trigger rasters

src/
        run_all.py # Complete Python pipeline

data/
        monthly_timeseries_sample.npz # Sample time series data
        coordinates_sample.csv # Sample coordinates
        events_sample.csv # Sample landslide events
