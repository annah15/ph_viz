# PH Vizualization

This project provides a visualization of different simplicial filtrations of a point cloud. 

## Setup
The demo requires Bokeh to be installed. 
Additionally, the Numpy and Gudhi library are required for this demo in order to run.
To install all packages in a virtual environment, run the following commands:

```bash
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Running the app
To view the app directly from a Bokeh server, stay in the parent directory ph-visualization, and execute the command:

```bash
bokeh serve --show filtration_viz.py
```