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
        return (f"Flight {self.callsign} with aircraft {self.icao24} on {t:%Y-%m-%d}")

    def avg_vrate(self):
        data = self.data.copy()
        if "vrate" in data.columns:
            s = pd.to_numeric(data["vrate"], errors="coerce")
        elif "vertical_rate" in data.columns:
            s = pd.to_numeric(data["vertical_rate"], errors="coerce")
        elif "bds60" in data.columns:
            s = data["bds60"].apply(lambda d: d.get("vrate_barometric") if isinstance(d, dict) else float("nan"))
            s = pd.to_numeric(s, errors="coerce")
        s = s.dropna()
        return float(s.mean())

    def net_altitude_change(self):
        alt = self.data.sort_values("timestamp")["altitude"].dropna()
        return float(alt.iloc[-1] - alt.iloc[0])

    def is_landing(self):
        return self.min("altitude") < 10900 and (self.avg_vrate() < -500 or self.net_altitude_change() < -500)

    def is_takeoff(self):
        return self.min("altitude") < 10900 and (self.avg_vrate() > 500 or self.net_altitude_change() > 500)

    def plot(self, ax, color, linewidth=1, alpha=1.0, zorder=3):
        data = self.data.copy()
        data["latitude"] = pd.to_numeric(data.get("latitude"), errors="coerce")
        data["longitude"] = pd.to_numeric(data.get("longitude"), errors="coerce")
        data = data.dropna(subset=["latitude", "longitude"]).sort_values("timestamp")
        data = data.drop_duplicates(subset=["timestamp", "latitude", "longitude"], keep="last")
        ax.plot(data["longitude"],data["latitude"],color=color,linewidth=linewidth, alpha=alpha, zorder=zorder,)

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
    gap = data["timestamp"].diff().dt.total_seconds()
    segment = (gap > max_gap_s).cumsum()
    for _, chunk in data.groupby(segment):
        yield chunk


class FlightCollection:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"FlightCollection with {len(self)} flights"

    COLONNES: list[str] = ["timestamp","icao24", "callsign", "latitude", "longitude","altitude","vertical_rate"]
    PETITE_COURONNE = ("75", "92", "93", "94")
    GRANDE_COURONNE = ("77", "78", "91", "95")
    EXTRA_AIRFIELDS = {"Toussus-le-Noble": (48.7519, 2.1061),"Etampes-Mondesir": (48.3822, 2.0653)}

    @classmethod
    def read_json(cls, *filenames):
        frames = []
        for filename in filenames:
            data= pd.read_json(filename, lines=True, dtype={"icao24": str})
            data= data[[c for c in cls.COLONNES if c in data.columns]]
            data["receiver"] = os.path.basename(filename)
            frames.append(data)
        data = pd.concat(frames, ignore_index=True).sort_values("timestamp")

        if "callsign" not in data.columns and "bds20" in data.columns:
            data["callsign"] = data["bds20"].apply(lambda d: d.get("callsign") if isinstance(d, dict) else None)

        if "callsign" in data.columns:
            data["callsign"] = data.groupby("icao24")["callsign"].transform(lambda s: s.ffill().bfill())

        if "vrate" not in data.columns:
            data["vrate"] = pd.to_numeric(data["vertical_rate"], errors="coerce")

        if {"latitude", "longitude"} <= set(data.columns):
            has_pos = data["latitude"].notna() & data["longitude"].notna()
        else:
            has_pos = pd.Series(False, index=data.index)

        has_vrate = data["vrate"].notna() if "vrate" in data.columns else pd.Series(False, index=data.index)

        data = data.loc[has_pos | has_vrate].copy()

        return cls(data)

    def avg_vrates(self):
        moy = []
        for _, flight in enumerate(self):
            vr = flight.avg_vrate()
            if pd.notna(vr):
                moy.append(vr)
        return pd.Series(moy, name="avg_vrate")

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


    @classmethod
    def plot_map(cls, ax):
        """Draw the Ile-de-France background on ax. Returns the outline of the departments."""
        idf = gpd.read_file(os.environ["URL_IDF"]).to_crs(epsg=4326)
        idf["CODE_DEPT"] = idf["CODE_DEPT"].astype(str)
        petite_couronne = idf[idf["CODE_DEPT"].isin(cls.PETITE_COURONNE)]
        grande_couronne = idf[idf["CODE_DEPT"].isin(cls.GRANDE_COURONNE)]
        airports = gpd.read_file(os.environ["URL_AIRPORTS"]).to_crs(epsg=4326)

        names = list(cls.EXTRA_AIRFIELDS)
        lats = [cls.EXTRA_AIRFIELDS[n][0] for n in names]
        lons = [cls.EXTRA_AIRFIELDS[n][1] for n in names]
        airfields = gpd.GeoDataFrame({"name": names}, geometry=gpd.points_from_xy(lons, lats), crs=4326)

        grande_couronne.plot(ax=ax, color="#D9D6D6",edgecolor="#6B6868", linewidth=0.3)
        petite_couronne.plot(ax=ax, color="#A29F9F",edgecolor="#0A0909", linewidth=0.5)
        airports.plot(ax=ax, color="#A29F9F",edgecolor="#EE0808", linewidth=0.5)
        airfields.plot(ax=ax, color="#EE0808", marker="^", markersize=25, zorder=5)
        for _, a in airfields.iterrows():
            ax.annotate(a["name"], (a.geometry.x, a.geometry.y), fontsize=7,xytext=(4, 4), textcoords="offset points", zorder=6)

        return pd.concat([petite_couronne, grande_couronne]).union_all()


def keep_inside_idf(data, boundary):
    points = gpd.GeoDataFrame(data[["icao24", "timestamp", "latitude", "longitude"]].copy(),geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),crs=4326,)
    return points[points.geometry.within(boundary)].copy()


def flight_min_altitudes(collection):
    data = collection.data[["icao24", "altitude"]].dropna().copy()
    data["altitude"] = pd.to_numeric(data["altitude"], errors="coerce")
    data = data.dropna(subset=["altitude"])
    mins = data.groupby("icao24", dropna=False)["altitude"].min()
    return mins


files =[os.environ["URL_Avions_MCD"]]  + [os.environ["URL_Avions_AT"]] + [os.environ["URL_Avions_AF"]] + [os.environ["URL_Avions_TC"]] + [os.environ["URL_Avions_R"]]
collection = FlightCollection.read_json(*files)

fig, ax = plt.subplots(figsize=(10, 6))

idf_boundary = FlightCollection.plot_map(ax)

ax.plot([], [], color="C0", label="Landing")
ax.plot([], [], color="C1", label="Takeoff")

n = 0
for flight in collection:
    if flight.is_landing():
        color = "C0"
    elif flight.is_takeoff():
        color = "C1"
    else:
        continue
    data = flight.data.copy()
    data = keep_inside_idf(data, idf_boundary)
    if data.empty:
        continue
    ax.plot(data["longitude"], data["latitude"], color=color, linewidth=1)

    n += 1
print(f"plotted {n} flights")
ax.legend()

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