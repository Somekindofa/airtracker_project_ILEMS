"""
Aircraft trajectories over the Petite Couronne (75, 92, 93, 94).

Classes
-------
BaseMap               : departments + airports layers (your existing code)
AdsbLoader            : reads the .jsonl and returns a clean positions DataFrame
Trajectory            : one aircraft (icao24), with its callsign and points
TrajectoryCollection  : all aircraft: iteration, summary, GeoDataFrame export
MapPlotter            : draws base map + airports + trajectories

.env must define URL_IDF, URL_AIRPORTS and URL_Avions_MCD.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from shapely.geometry import LineString

load_dotenv()


# --------------------------------------------------------------------------- #
# Base map
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Airfield:
    """A named airfield drawn as a marker (for airfields missing from your shapefile)."""
    name: str
    lat: float
    lon: float


class BaseMap:
    """Petite + Grande Couronne, airports and extra airfields, all in one CRS."""

    PETITE_COURONNE = ("75", "92", "93", "94")
    GRANDE_COURONNE = ("77", "78", "91", "95")

    # West of Orly (approx. airfield reference points)
    DEFAULT_AIRFIELDS = (
        Airfield("Villacoublay", 48.7743, 2.2019),
        Airfield("Toussus-le-Noble", 48.7519, 2.1061),
    )

    def __init__(self, idf_path: str, airports_path: str, epsg: int = 4326,
                 include_grande_couronne: bool = True,
                 airfields: tuple[Airfield, ...] = DEFAULT_AIRFIELDS):
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
        self.airfields = gpd.GeoDataFrame(
            {"name": [a.name for a in airfields]},
            geometry=gpd.points_from_xy([a.lon for a in airfields],
                                        [a.lat for a in airfields]),
            crs="EPSG:4326",
        ).to_crs(epsg=epsg)

    @classmethod
    def from_env(cls, epsg: int = 4326, **kwargs) -> "BaseMap":
        return cls(os.environ["URL_IDF"], os.environ["URL_AIRPORTS"], epsg, **kwargs)

    @property
    def bounds(self):
        return self.departments.total_bounds  # minx, miny, maxx, maxy

    @property
    def petite_couronne_bounds(self):
        return self.petite_couronne.total_bounds


# --------------------------------------------------------------------------- #
# CPR decoding (airborne positions, 17-bit)
# --------------------------------------------------------------------------- #
class CprDecoder:
    """
    Local CPR decoding: lat/lon from ONE message + a reference position
    (valid within ~300 km of the reference, fine for a receiver in Paris).
    Used to recover the positions that the file left undecoded.
    """

    NZ = 15

    def __init__(self, ref_lat: float, ref_lon: float):
        self.ref_lat, self.ref_lon = ref_lat, ref_lon

    @classmethod
    def _nl(cls, lat: float) -> int:
        if abs(lat) >= 87:
            return 2
        a = 1 - math.cos(math.pi / (2 * cls.NZ))
        b = math.cos(math.pi * abs(lat) / 180) ** 2
        return int(math.floor(2 * math.pi / math.acos(1 - a / b)))

    def decode(self, lat_cpr: float, lon_cpr: float, odd: bool) -> tuple[float, float]:
        i = 1 if odd else 0
        lat_c, lon_c = lat_cpr / 131072, lon_cpr / 131072

        d_lat = 360 / (4 * self.NZ - i)
        j = math.floor(self.ref_lat / d_lat) + math.floor(
            0.5 + (self.ref_lat % d_lat) / d_lat - lat_c)
        lat = d_lat * (j + lat_c)

        d_lon = 360 / max(self._nl(lat) - i, 1)
        m = math.floor(self.ref_lon / d_lon) + math.floor(
            0.5 + (self.ref_lon % d_lon) / d_lon - lon_c)
        return lat, d_lon * (m + lon_c)


# --------------------------------------------------------------------------- #
# ADS-B loading
# --------------------------------------------------------------------------- #
class AdsbLoader:
    """
    Position messages (latitude/longitude) do NOT contain the callsign:
    it is sent in separate identification messages. So we group by icao24
    and attach the callsign afterwards.
    """

    def __init__(self, path: str, reference: tuple[float, float] = (48.80, 2.33),
                 decode_missing: bool = True):
        self.path = path
        self.decoder = CprDecoder(*reference)  # reference ~ receiver (Arcueil)
        self.decode_missing = decode_missing

    def _fill_missing_positions(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Rows with raw CPR but no decoded lat/lon -> decode them locally."""
        todo = raw["lat_cpr"].notna() & raw["latitude"].isna()
        for idx, r in raw[todo].iterrows():
            lat, lon = self.decoder.decode(r["lat_cpr"], r["lon_cpr"], r["parity"] == "odd")
            raw.at[idx, "latitude"], raw.at[idx, "longitude"] = lat, lon
        print(f"{todo.sum()} positions recovered from raw CPR")
        return raw

    def load(self) -> pd.DataFrame:
        raw = pd.read_json(self.path, lines=True, convert_dates=False)
        if self.decode_missing:
            raw = self._fill_missing_positions(raw)

        # icao24 -> callsign (last non-null value seen)
        callsigns = (
            raw.dropna(subset=["callsign"])
            .assign(callsign=lambda d: d["callsign"].str.strip().str.rstrip("_"))
            .drop_duplicates("icao24", keep="last")
            .set_index("icao24")["callsign"]
        )

        pos = raw.dropna(subset=["latitude", "longitude"]).copy()
        pos["time"] = pd.to_datetime(pos["timestamp"], unit="s", utc=True)
        pos["callsign"] = pos["icao24"].map(callsigns)
        pos["label"] = pos["callsign"].fillna(pos["icao24"])

        cols = ["icao24", "callsign", "label", "timestamp", "time",
                "latitude", "longitude", "altitude"]
        return (
            pos[cols]
            .sort_values(["icao24", "timestamp"])
            .drop_duplicates(["icao24", "latitude", "longitude"])
            .reset_index(drop=True)
        )


# --------------------------------------------------------------------------- #
# One aircraft
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    icao24: str
    data: pd.DataFrame  # sorted by timestamp

    @property
    def callsign(self) -> Optional[str]:
        v = self.data["callsign"].iloc[0]
        return None if pd.isna(v) else v

    @property
    def label(self) -> str:
        return self.callsign or self.icao24

    @property
    def start(self) -> pd.Timestamp:
        return self.data["time"].min()

    @property
    def end(self) -> pd.Timestamp:
        return self.data["time"].max()

    @property
    def max_altitude(self) -> Optional[float]:
        a = self.data["altitude"].max()
        return None if pd.isna(a) else a

    @property
    def min_altitude(self) -> Optional[float]:
        a = self.data["altitude"].min()  # NaN ignored
        return None if pd.isna(a) else a

    def __len__(self) -> int:
        return len(self.data)

    def segments(self, max_gap_s: Optional[float] = None) -> list[pd.DataFrame]:
        """Split the track where reception stopped for more than max_gap_s."""
        if max_gap_s is None:
            return [self.data]
        new_seg = self.data["timestamp"].diff().gt(max_gap_s).cumsum()
        return [seg for _, seg in self.data.groupby(new_seg)]

    def to_linestrings(self, max_gap_s: Optional[float] = None) -> list[LineString]:
        return [
            LineString(zip(seg["longitude"], seg["latitude"]))
            for seg in self.segments(max_gap_s)
            if len(seg) >= 2
        ]

    def __repr__(self) -> str:
        return f"Trajectory({self.label!r}, {len(self)} pts)"


# --------------------------------------------------------------------------- #
# All aircraft
# --------------------------------------------------------------------------- #
class TrajectoryCollection:
    def __init__(self, trajectories: list[Trajectory]):
        self._trajs = trajectories

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TrajectoryCollection":
        return cls([Trajectory(icao, chunk.reset_index(drop=True))
                    for icao, chunk in df.groupby("icao24")])

    def __len__(self) -> int:
        return len(self._trajs)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._trajs)

    def filter_min_points(self, n: int) -> "TrajectoryCollection":
        return TrajectoryCollection([t for t in self._trajs if len(t) >= n])

    def filter_low_altitude(self, threshold_ft: float = 3000) -> "TrajectoryCollection":
        """Keep aircraft that go below threshold_ft at some point
        (= probable takeoffs / landings). The whole track is kept."""
        return TrajectoryCollection(
            [t for t in self._trajs
             if t.min_altitude is not None and t.min_altitude < threshold_ft]
        )

    def crop_to_bbox(self, minx, miny, maxx, maxy) -> "TrajectoryCollection":
        """Keep only the position points inside a lon/lat box (removes outliers)."""
        out = []
        for t in self._trajs:
            d = t.data
            d = d[d["longitude"].between(minx, maxx) & d["latitude"].between(miny, maxy)]
            if len(d):
                out.append(Trajectory(t.icao24, d.reset_index(drop=True)))
        return TrajectoryCollection(out)

    def summary(self) -> pd.DataFrame:
        """Replaces your groupby('callsign').agg(...) -- but on icao24."""
        return pd.DataFrame(
            [{"icao24": t.icao24, "callsign": t.callsign, "count": len(t),
              "min": t.start, "max": t.end, "min_alt": t.min_altitude, "max_alt": t.max_altitude}
             for t in self._trajs]
        ).set_index("icao24")

    def to_geodataframe(self, max_gap_s: Optional[float] = None,
                        epsg: int = 4326) -> gpd.GeoDataFrame:
        rows = [
            {"icao24": t.icao24, "callsign": t.label, "min_alt": t.min_altitude,
             "max_alt": t.max_altitude, "geometry": line}
            for t in self._trajs
            for line in t.to_linestrings(max_gap_s)
        ]
        return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(epsg=epsg)

    def save_shapefile(self, path: str, max_gap_s: Optional[float] = None,
                       epsg: int = 4326) -> None:
        self.to_geodataframe(max_gap_s, epsg).to_file(path, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
class MapPlotter:
    def __init__(self, basemap: BaseMap, figsize=(8, 8)):
        self.basemap = basemap
        self.figsize = figsize

    def diagnose(self, trajectories: TrajectoryCollection) -> None:
        """Compare the extents (lon/lat) of the base map and the trajectories."""
        g = trajectories.to_geodataframe(epsg=self.basemap.epsg)
        print("basemap      [minx miny maxx maxy]:", self.basemap.bounds.round(3))
        print("trajectories [minx miny maxx maxy]:", g.total_bounds.round(3))
        print("basemap CRS:", self.basemap.departments.crs)

    def plot(self, trajectories: TrajectoryCollection,
             max_gap_s: Optional[float] = None,
             zoom_on: str = "basemap",   # "basemap" | "petite_couronne" | "trajectories"
             color_by_altitude: bool = True,
             color_column: str = "min_alt",   # "min_alt" or "max_alt"
             labels: bool = False):
        fig, ax = plt.subplots(figsize=self.figsize)

        bm = self.basemap
        if len(bm.grande_couronne):
            bm.grande_couronne.plot(ax=ax, color="#D9D6D6",
                                    edgecolor="#6B6868", linewidth=0.3)
        bm.petite_couronne.plot(ax=ax, color="#A29F9F",
                                edgecolor="#0A0909", linewidth=0.5)
        bm.airports.plot(ax=ax, color="#A29F9F", edgecolor="#EE0808", linewidth=0.5)

        # extra airfields (Villacoublay, Toussus...) as labelled red markers
        bm.airfields.plot(ax=ax, color="#EE0808", marker="^", markersize=30, zorder=5)
        for _, a in bm.airfields.iterrows():
            ax.annotate(a["name"], (a.geometry.x, a.geometry.y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", zorder=6)

        gdf = trajectories.to_geodataframe(max_gap_s, self.basemap.epsg)
        if color_by_altitude:
            gdf.plot(ax=ax, column=color_column, cmap="viridis", linewidth=1,
                     legend=True, legend_kwds={"label": f"{color_column} (ft)",
                                               "shrink": 0.6}, zorder=3)
        else:
            gdf.plot(ax=ax, color="#1f77b4", linewidth=1, zorder=3)

        if labels:
            for _, r in gdf.iterrows():
                x, y = r.geometry.coords[-1]
                ax.annotate(r["callsign"], (x, y), fontsize=6, zorder=4)

        # Most flights are outside the Petite Couronne: choose the zoom
        if zoom_on == "basemap":
            minx, miny, maxx, maxy = self.basemap.bounds
        elif zoom_on == "petite_couronne":
            minx, miny, maxx, maxy = self.basemap.petite_couronne_bounds
        else:
            minx, miny, maxx, maxy = gdf.total_bounds
        mx, my = (maxx - minx) * 0.03, (maxy - miny) * 0.03
        ax.set_xlim(minx - mx, maxx + mx)
        ax.set_ylim(miny - my, maxy + my)
        ax.set_axis_off()
        return fig, ax


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    basemap = BaseMap.from_env()
    positions = AdsbLoader(os.environ["URL_Avions_MCD"]).load()

    trajs = TrajectoryCollection.from_dataframe(positions).filter_min_points(5)
    print(trajs.summary())

    plotter = MapPlotter(basemap)
    plotter.diagnose(trajs)

    # keep only what is around the Petite Couronne (drops far-away / wrong points)
    minx, miny, maxx, maxy = basemap.bounds
    pad = 0.3  # degrees
    local = trajs.crop_to_bbox(minx - pad, miny - pad, maxx + pad, maxy + pad)
    print(len(local), "aircraft near the Petite Couronne")

    # only takeoffs / landings: below 3000 ft at some point
    local = local.filter_low_altitude(3000)
    print(len(local), "aircraft below 3000 ft (takeoff / landing)")
    print(local.summary())

    plotter.plot(local, max_gap_s=120, zoom_on="basemap")
    plt.show()