import os
from datetime import timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class Flight:

    def __lt__(a, b):
        return a.min("timestamp") <= b.min("timestamp")

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        t = pd.to_datetime(self.min("timestamp"), unit="s")
        return (
            f"Flight {self.callsign} with aircraft {self.icao24} "
            f"on {t:%Y-%m-%d}"
        )

    def avg_vrate(self):
        df = self.data.copy()

        if "vrate" in df.columns:
            s = pd.to_numeric(df["vrate"], errors="coerce")
        elif "vertical_rate" in df.columns:
            s = pd.to_numeric(df["vertical_rate"], errors="coerce")
        elif "bds60" in df.columns:
            s = df["bds60"].apply(
                lambda d: d.get("vrate_barometric") if isinstance(d, dict) else float("nan")
            )
            s = pd.to_numeric(s, errors="coerce")
        else:
            return float("nan")

        s = s.dropna()
        if s.empty:
            return float("nan")
        return float(s.mean())

    def plot(self, ax, color, linewidth=1, alpha=1.0, zorder=3):
        data = self.data.copy()
        data["latitude"] = pd.to_numeric(data.get("latitude"), errors="coerce")
        data["longitude"] = pd.to_numeric(data.get("longitude"), errors="coerce")
        data = data.dropna(subset=["latitude", "longitude"]).sort_values("timestamp")
        data = data.drop_duplicates(subset=["timestamp", "latitude", "longitude"], keep="last")
        if data.empty:
            return
        ax.plot(
            data["longitude"],
            data["latitude"],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

    def max(self, feature):
        return self.data[feature].max()

    def min(self, feature):
        return self.data[feature].min()

    @property
    def callsign(self):
        return self.min("callsign")

    @property
    def icao24(self):
        return self.min("icao24")


def iterate_icao24_callsign(data):
    for _, group in data.groupby(["icao24", "callsign"], dropna=False):
        yield group


def iterate_time(data, max_gap_s):
    data = data.sort_values("timestamp")
    gap = data["timestamp"].diff()
    if pd.api.types.is_timedelta64_dtype(gap):
        threshold = pd.Timedelta(seconds=max_gap_s)
    else:
        threshold = max_gap_s
    segment = (gap > threshold).cumsum()
    for _, chunk in data.groupby(segment):
        yield chunk


class FlightCollection:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"FlightCollection with {len(self)} flights"

    @classmethod
    def read_json(cls, filename):
        data = pd.read_json(filename, lines=True)
        if "callsign" not in data.columns and "bds20" in data.columns:
            data["callsign"] = data["bds20"].apply(
                lambda d: d.get("callsign") if isinstance(d, dict) else None
            )

        # Build a unified vrate column from available sources
        if "vrate" not in data.columns:
            if "vertical_rate" in data.columns:
                data["vrate"] = pd.to_numeric(data["vertical_rate"], errors="coerce")
            elif "bds60" in data.columns:
                def _extract_vrate(x):
                    if isinstance(x, dict):
                        return x.get("vrate_barometric", float("nan"))
                    return float("nan")
                data["vrate"] = data["bds60"].apply(_extract_vrate)
                data["vrate"] = pd.to_numeric(data["vrate"], errors="coerce")

        # Keep rows that have either a valid position OR a vrate value
        if {"latitude", "longitude"} <= set(data.columns):
            has_pos = data["latitude"].notna() & data["longitude"].notna()
        else:
            has_pos = pd.Series(False, index=data.index)

        has_vrate = data["vrate"].notna() if "vrate" in data.columns else pd.Series(False, index=data.index)

        data = data.loc[has_pos | has_vrate].copy()

        return cls(data)

    def avg_vrates(self):
        # Return one average vertical rate value per Flight segment (not aggregated by icao24)
        vals = []
        for i, flight in enumerate(self):
            vr = flight.avg_vrate()
            if pd.notna(vr):
                vals.append(vr)
        return pd.Series(vals, name="avg_vrate")

    def __iter__(self):
        for group in iterate_icao24_callsign(self.data):
            for elt in iterate_time(group, 20000):
                yield Flight(elt)

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, key):
        if isinstance(key, str):
            result = FlightCollection(
                self.data.query("callsign == @key or icao24 == @key")
            )
        elif isinstance(key, pd.Timestamp):
            before = key.timestamp()
            after = (key + timedelta(days=1)).timestamp()
            result = FlightCollection(
                self.data[(self.data["timestamp"] >= before) &
                          (self.data["timestamp"] < after)]
            )
        else:
            raise KeyError(key)

        if len(result) == 1:
            return Flight(result.data)
        return result


class BaseMap:
    PETITE_COURONNE = ("75", "92", "93", "94")
    GRANDE_COURONNE = ("77", "78", "91", "95")

    def __init__(self, idf_path, airports_path, epsg=4326,
                 include_grande_couronne=True):
        self.epsg = epsg
        idf = gpd.read_file(idf_path).to_crs(epsg=epsg)
        idf["CODE_DEPT"] = idf["CODE_DEPT"].astype(str)
        self.petite_couronne = idf[idf["CODE_DEPT"].isin(self.PETITE_COURONNE)]
        self.grande_couronne = (
            idf[idf["CODE_DEPT"].isin(self.GRANDE_COURONNE)]
            if include_grande_couronne else idf.iloc[0:0]
        )
        self.departments = pd.concat([self.petite_couronne, self.grande_couronne])
        self.airports = gpd.read_file(airports_path).to_crs(epsg=epsg)

    @classmethod
    def from_env(cls, epsg=4326, **kwargs):
        return cls(os.environ["URL_IDF"], os.environ["URL_AIRPORTS"], epsg, **kwargs)

    @property
    def bounds(self):
        return self.departments.total_bounds

    @property
    def petite_couronne_bounds(self):
        return self.petite_couronne.total_bounds


def keep_inside_idf(data, boundary):
    if data.empty:
        return data
    points = gpd.GeoDataFrame(
        data[["icao24", "timestamp", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
        crs=4326,
    )
    return points[points.geometry.within(boundary)].copy()


def flight_min_altitudes(collection):
    df = collection.data[["icao24", "altitude"]].dropna().copy()
    if df.empty:
        return pd.Series(dtype=float)

    df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce")
    df = df.dropna(subset=["altitude"])
    mins = df.groupby("icao24", dropna=False)["altitude"].min()
    return mins


if __name__ == "__main__":
    basemap = BaseMap.from_env()
    collection = FlightCollection.read_json(os.environ["URL_Avions_MCD"])

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(basemap.grande_couronne):
        basemap.grande_couronne.plot(ax=ax, color="#D9D6D6",
                                     edgecolor="#6B6868", linewidth=0.3)
    basemap.petite_couronne.plot(ax=ax, color="#A29F9F",
                                 edgecolor="#0A0909", linewidth=0.5)
    basemap.airports.plot(ax=ax, color="#A29F9F",
                          edgecolor="#EE0808", linewidth=0.5)

    idf_boundary = basemap.departments.unary_union

    n = 0
    for flight in collection:
        data = flight.data.copy()
        data = keep_inside_idf(data, idf_boundary)
        if data.empty:
            continue
        ax.plot(data["longitude"], data["latitude"], color="C0", linewidth=1)
        n += 1
    print(f"plotted {n} flights")

    mins = flight_min_altitudes(collection)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(mins.values, bins=40, edgecolor="black")
    ax2.set_title("Histogram of minimum altitude per flight")
    ax2.set_xlabel("Minimum altitude (ft)")
    ax2.set_ylabel("Number of flights")

    q1, q2, q3 = list(mins.quantile([0.25, 0.5, 0.75]))

    ax2.axvline(q1, color="red", linestyle="--", label=f"Q1 = {q1:.0f} ft")
    ax2.axvline(q2, color="orange", linestyle="--", label=f"Median = {q2:.0f} ft")
    ax2.axvline(q3, color="green", linestyle="--", label=f"Q3 = {q3:.0f} ft")
    ax2.legend()

    plt.show()

    vrates = collection.avg_vrates()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(vrates.values, bins=40, edgecolor="black")
    ax3.set_title("Average vertical rate per flight")
    ax3.set_xlabel("Average vertical rate (ft/min)")
    ax3.set_ylabel("Number of flights")
    plt.show()