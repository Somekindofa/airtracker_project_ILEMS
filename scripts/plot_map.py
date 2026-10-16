import os
import geopandas as gpd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from matplotlib.axes import Axes


def load_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    load_dotenv()
    idf = gpd.read_file(os.environ["URL_IDF"]).query("CODE_DEPT in ['94', '93', '92', '75']")
    runways = gpd.read_file(os.environ["URL_RUNWAYS"])
    idf = idf.to_crs(epsg=4326)
    runways = runways.to_crs(epsg=4326)
    return idf, runways


def plot_map(idf: gpd.GeoDataFrame, runways: gpd.GeoDataFrame) -> Axes:
    _fig, ax = plt.subplots(figsize=(8, 8))
    idf.plot(ax=ax, color="#A29F9F", edgecolor="#0A0909", linewidth=0.5)
    # Runways are line geometry, not polygons, so `color` controls the line
    # itself — `edgecolor` has no effect here.
    runways.plot(ax=ax, color="#EE0808", linewidth=1)
    ax.set_axis_off()
    return ax


if __name__ == "__main__":
    idf, runways = load_layers()
    plot_map(idf, runways)
    plt.show()
