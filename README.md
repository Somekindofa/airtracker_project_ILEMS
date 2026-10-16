# airtracker_project_ILEMS

## Setup (Python 3.13)

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then create a `.env` file next to `project.ipynb` with the local paths of the shapefiles:

```
URL_IDF = "C:\\path\\to\\geoflar-departements.shp"
URL_AIRPORTS = "C:\\path\\to\\aeroports-dile-de-france0.shp"
```

In the notebook, select the `.venv` kernel.
