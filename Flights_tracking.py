import os
import geopandas as gpd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
gdf: gpd.GeoDataFrame = gpd.read_file(os.environ["URL_IDF"]).query("CODE_DEPT in ['94', '93', '92', '75']")
gdf_airports: gpd.GeoDataFrame = gpd.read_file(os.environ["URL_AIRPORTS"])

gdf = gdf.to_crs(epsg=4326)
gdf_airports = gdf_airports.to_crs(epsg=4326)

fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, color="#A29F9F", edgecolor="#0A0909", linewidth=0.5)
gdf_airports.plot(ax=ax, color="#0C4CB3", linewidth=0.9)
ax.set_axis_off()
plt.show()