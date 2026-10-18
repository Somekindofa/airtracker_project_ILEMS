import argparse
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.crs import EuroPP, PlateCarree
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

CACHE = "data/flights_clean.pkl.gz"
START_UTC = pd.Timestamp("2026-10-15 13:00")
END_UTC = pd.Timestamp("2026-10-15 14:00")
GAP_S = 300  # silence (s) after which a new flight starts
LOW_FT = 500  # a flight has to go below this altitude
CUT_FT = 1500  # |altitude change| above this = landing / take off, below = low flyer
MAX_JOIN_S = 60  # no line between two points further apart than this
VMAX_FT = 12000  # everything above is just blue


def load_positions(cache=CACHE):
    """Rows of the capture hour that have altitude, latitude and longitude, sorted by time."""
    cols = ["timestamp", "icao24", "altitude", "latitude", "longitude"]
    data = pd.read_pickle(cache)[cols]  # our own cache, built from the group files (pickle is fine here)

    # the clock is UTC; this also drops Chatou (recorded the evening before)
    data = data[(data["timestamp"] >= START_UTC) & (data["timestamp"] < END_UTC)].copy()
    data["timestamp"] = data["timestamp"] + pd.Timedelta(hours=2)

    return (
        data.query("0 < altitude < 45000 & longitude.notna() & latitude.notna()")
        .sort_values(["icao24", "timestamp"])
        .reset_index(drop=True)
    )


def split_flights(pos):
    """Add a flight_id column: new flight when the aircraft changes or goes silent for GAP_S."""
    new = (pos["icao24"] != pos["icao24"].shift()) | (pos["timestamp"].diff().dt.total_seconds() > GAP_S)
    return pos.assign(flight_id=new.cumsum())


def select_landings_takeoffs(pos):
    """Return the flight ids of the landings and of the take offs."""
    alt = pos.groupby("flight_id")["altitude"]
    went_low = alt.min() < LOW_FT
    change = alt.last() - alt.first()  # rows are sorted by time inside a flight
    landings = change.index[went_low & (change < -CUT_FT)]
    takeoffs = change.index[went_low & (change > CUT_FT)]
    return landings, takeoffs


def plot(pos, flight_ids, title, idf, runways):
    norm = Normalize(vmin=0, vmax=VMAX_FT)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": EuroPP()})

    idf.plot(ax=ax, transform=PlateCarree(), color="#A29F9F", edgecolor="#0A0909", linewidth=0.5)
    runways.plot(ax=ax, transform=PlateCarree(), color="#EE0808", linewidth=1)

    for _, d in pos[pos["flight_id"].isin(flight_ids)].groupby("flight_id"):
        pts = np.column_stack([d["longitude"], d["latitude"]]).reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        alt = d["altitude"].to_numpy()
        seg_alt = (alt[:-1] + alt[1:]) / 2  # one altitude per segment
        close = d["timestamp"].diff().dt.total_seconds().to_numpy()[1:] < MAX_JOIN_S
        lc = LineCollection(segments[close], cmap="RdYlBu", norm=norm, transform=PlateCarree(), linewidth=1.5)
        lc.set_array(seg_alt[close])
        ax.add_collection(lc)

    fig.colorbar(lc, ax=ax, shrink=0.6, label="Altitude (ft)")
    minx, miny, maxx, maxy = idf.total_bounds
    ax.set_extent((minx, maxx, miny, maxy), crs=PlateCarree())
    ax.set_title(title)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="output/project_theo.png")
    args = ap.parse_args()

    pos = split_flights(load_positions())
    landings, takeoffs = select_landings_takeoffs(pos)
    print(f"{pos['flight_id'].nunique()} flights, {len(landings)} landings, {len(takeoffs)} take offs")

    idf = gpd.read_file("geoflar-departements/geoflar-departements.shp").to_crs(epsg=4326)
    runways = gpd.read_file("aeroports-dile-de-france/export_runways.geojson").to_crs(epsg=4326)

    title = f"{len(landings)} landings and {len(takeoffs)} take offs, 3pm to 4pm"
    fig = plot(pos, landings.union(takeoffs), title, idf, runways)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
