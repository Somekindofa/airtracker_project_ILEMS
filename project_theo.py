import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.crs import EuroPP, PlateCarree
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

CACHE = "data/flights_clean.pkl.gz"             # so that we don'tt reload the whole db evertime
OUT = "output/landing_takeoff.png"              # final png of the landings and take offs
START_UTC = pd.Timestamp("2026-10-15 13:00")    # the UTC is UTC+0 in the dataset... we'll convert later
END_UTC = pd.Timestamp("2026-10-15 14:00")
GAP_S = 300                                     # silence (s) after which a new flight starts
LOW_FT = 400                                    # a flight has to go below this altitude
CUT_FT = 1500                                   # above this = landing or take off, below = low flyer
MAX_JOIN_S = 60                                 # no line between two points further apart than this
VMAX_FT = 12000                                 # everything above is just blue (better visuals)


def iterate_time(data: pd.DataFrame, threshold):
    idx = np.where(data.timestamp.diff().dt.total_seconds() > threshold)[0]
    start = 0
    for stop in idx:
        yield data.iloc[start:stop]
        start = stop
    yield data.iloc[start:]


class Flight:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __repr__(self):
        return f"Flight with icao24 {self.icao24} and {len(self.data)} rows"

    def max(self, feature):
        return self.data[feature].max()

    def min(self, feature):
        return self.data[feature].min()

    @property
    def icao24(self):
        return self.min("icao24")


class FlightCollection:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __repr__(self):
        return f"FlightCollection with {len(self)} flights"

    @classmethod
    def read_pickle(cls, pkl):
        # our own pickel cache, built from the group files
        return cls(pd.read_pickle(pkl))

    def __iter__(self):
        for _, group in self.data.groupby(by="icao24"):
            for elt in iterate_time(group, GAP_S):
                yield Flight(elt)

    def __len__(self):
        return sum(1 for _ in self)


def load_flights(cache=CACHE):
    """FlightCollection of the capture hour, rows that have altitude, latitude and longitude."""
    data = FlightCollection.read_pickle(cache).data

    # the clock is UTC; this also drops Chatou (recorded the evening before)
    data = data[(data["timestamp"] >= START_UTC) & (data["timestamp"] < END_UTC)].copy()
    data["timestamp"] = data["timestamp"] + pd.Timedelta(hours=2)

    data = data.query("0 < altitude < 45000 & longitude.notna() & latitude.notna()").sort_values("timestamp")
    return FlightCollection(data)


def select_landings_takeoffs(fc):
    """Return the landings and the take offs, as lists of Flight."""
    landings, takeoffs = [], []
    for flight in fc:
        if flight.min("altitude") >= LOW_FT:
            continue
        alt = flight.data["altitude"]  # rows are sorted by time inside a flight
        change = alt.iloc[-1] - alt.iloc[0]
        if change < -CUT_FT:
            landings.append(flight)
        elif change > CUT_FT:
            takeoffs.append(flight)
    return landings, takeoffs


def plot(flights, title, idf, runways):
    norm = Normalize(vmin=0, vmax=VMAX_FT)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": EuroPP()})

    idf.plot(ax=ax, transform=PlateCarree(), color="#A29F9F", edgecolor="#0A0909", linewidth=0.5)
    runways.plot(ax=ax, transform=PlateCarree(), color="#EE0808", linewidth=1)

    for flight in flights:
        d = flight.data
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
    ax.set_aspect("equal")  # geopandas' plot() sets 1/cos(lat) on lon/lat data, which stretches the map in EuroPP
    ax.set_title(title)
    return fig


def main():
    fc = load_flights()
    landings, takeoffs = select_landings_takeoffs(fc)
    print(f"{len(fc)} flights, {len(landings)} landings, {len(takeoffs)} take offs")

    idf = gpd.read_file("geoflar-departements/geoflar-departements.shp").to_crs(epsg=4326)
    runways = gpd.read_file("aeroports-dile-de-france/export_runways.geojson").to_crs(epsg=4326)

    title = f"{len(landings)} landings and {len(takeoffs)} take offs, 3pm to 4pm"
    fig = plot(landings + takeoffs, title, idf, runways)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
