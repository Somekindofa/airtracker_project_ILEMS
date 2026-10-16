"""
Aircraft trajectories over the Petite Couronne (75, 92, 93, 94).

Classes
-------
BaseMap               : departments + airports layers
AdsbLoader            : reads the .jsonl and returns a clean positions DataFrame
Flights               : one aircraft (icao24), with its callsign, points, vrate
FlightsCollection     : all aircraft: iteration, summary, GeoDataFrame export
MapPlotter            : draws base map + airports + trajectories + histograms

.env must define URL_IDF, URL_AIRPORTS and URL_Avions_MCD.
"""
from __future__ import annotations

import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from shapely.geometry import LineString

load_dotenv()


@dataclass(frozen=True)
class Airfield:
    """A named airfield drawn as a marker (for airfields missing from your shapefile)."""
    name: str
    lat: float
    lon: float


class BaseMap:
    """Petite + Grande Couronne, airports and extra airfields."""

    PETITE_COURONNE = ("75", "92", "93", "94")
    GRANDE_COURONNE = ("77", "78", "91", "95")
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
    Position messages (lat/lon) and vertical-rate messages arrive in separate
    rows. We group by icao24, attach the callsign, and merge the nearest vrate
    sample onto every position.
    """
    VRATE_SOURCES = ("vertical_rate", "vrate_barometric", "vrate_inertial")

    def __init__(self, path: str, reference: tuple[float, float] = (48.80, 2.33),
                 decode_missing: bool = True, vrate_tolerance_s: float = 2.0):
        self.path = path
        self.decoder = CprDecoder(*reference)
        self.decode_missing = decode_missing
        self.vrate_tolerance_s = vrate_tolerance_s

    def _fill_missing_positions(self, raw: pd.DataFrame) -> pd.DataFrame:
        todo = raw["lat_cpr"].notna() & raw["latitude"].isna()
        for idx, r in raw[todo].iterrows():
            lat, lon = self.decoder.decode(r["lat_cpr"], r["lon_cpr"], r["parity"] == "odd")
            raw.at[idx, "latitude"], raw.at[idx, "longitude"] = lat, lon
        print(f"{todo.sum()} positions recovered from raw CPR")
        return raw

    def _coalesce_vrate(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Merge the several vertical-rate sources into a single 'vrate' column."""
        for col in self.VRATE_SOURCES:
            if col not in raw.columns:
                raw[col] = pd.NA
        vrate = raw[self.VRATE_SOURCES[0]]
        for col in self.VRATE_SOURCES[1:]:
            vrate = vrate.fillna(raw[col])
        raw["vrate"] = pd.to_numeric(vrate, errors="coerce")
        return raw

    def load(self) -> pd.DataFrame:
        raw = pd.read_json(self.path, lines=True, convert_dates=False)
        if self.decode_missing:
            raw = self._fill_missing_positions(raw)
        raw = self._coalesce_vrate(raw)

        # ---- callsign per icao24 ------------------------------------------
        callsigns = (
            raw.dropna(subset=["callsign"])
            .assign(callsign=lambda d: d["callsign"].str.strip().str.rstrip("_"))
            .drop_duplicates("icao24", keep="last")
            .set_index("icao24")["callsign"]
        )

        # ---- position rows ------------------------------------------------
        pos = raw.dropna(subset=["latitude", "longitude"]).copy()
        # The position rows do NOT carry a real vertical rate (it lives in
        # other message types), so drop the placeholder column added by
        # _coalesce_vrate before merging; otherwise the merge renames both
        # to vrate_x / vrate_y.
        pos = pos.drop(columns=["vrate"], errors="ignore")
        pos["time"] = pd.to_datetime(pos["timestamp"], unit="s", utc=True)
        pos["callsign"] = pos["icao24"].map(callsigns)
        pos["label"] = pos["callsign"].fillna(pos["icao24"])

        # merge_asof requires the `on` key to be globally sorted.
        pos = pos.sort_values("timestamp").reset_index(drop=True)

        # ---- nearest vrate sample attached to each position ---------------
        vr = (
            raw.dropna(subset=["vrate"])[["icao24", "timestamp", "vrate"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(vr):
            pos = pd.merge_asof(
                pos, vr,
                on="timestamp", by="icao24",
                direction="nearest", tolerance=self.vrate_tolerance_s,
            )
        else:
            pos["vrate"] = pd.NA

        cols = ["icao24", "callsign", "label", "timestamp", "time",
                "latitude", "longitude", "altitude", "vrate"]
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
class Flights:
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

    # ------------------------------------------------------------------ NEW --
    def mean_vrate(self, altitude_max: Optional[float] = None) -> Optional[float]:
        """Mean vertical rate over the flight (ft/min).

        If `altitude_max` is given, only samples at or below that altitude are
        used — useful to focus on the takeoff/landing phase rather than on
        cruise where vrate ≈ 0 by construction.
        """
        if "vrate" not in self.data.columns:
            return None
        d = self.data.dropna(subset=["vrate"])
        if altitude_max is not None:
            d = d[d["altitude"] <= altitude_max]
        if not len(d):
            return None
        return float(d["vrate"].mean())

    def phase(self,
              takeoff_min: float = 200.0,
              landing_max: float = -200.0,
              altitude_max: Optional[float] = None,
              fallback_ground_ft: float = 3000.0,
              fallback_delta_ft: float = 1500.0) -> str:
        """Classify the flight as 'takeoff', 'landing' or 'cruising'.

        Primary rule (vrate-based): compare the mean vertical rate of the
        flight to the two thresholds `takeoff_min` and `landing_max` that
        the user reads off the vrate histogram.

        Fallback (altitude-delta) is used only when no vrate sample is
        available for the flight.
        """
        m = self.mean_vrate(altitude_max=altitude_max)
        if m is not None:
            if m >= takeoff_min:
                return "takeoff"
            if m <= landing_max:
                return "landing"
            return "cruising"

        # ---- fallback: altitude-delta --------------------------------
        alt = self.data["altitude"].dropna()
        if len(alt) < 3:
            return "cruising"
        k = max(2, len(alt) // 10)
        start = float(alt.head(k).median())
        end = float(alt.tail(k).median())
        delta = end - start
        if start <= fallback_ground_ft and delta >= fallback_delta_ft:
            return "takeoff"
        if end <= fallback_ground_ft and -delta >= fallback_delta_ft:
            return "landing"
        return "cruising"
    # ------------------------------------------------------------------------

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
        return f"Flights({self.label!r}, {len(self)} pts)"


# --------------------------------------------------------------------------- #
# All aircraft
# --------------------------------------------------------------------------- #
class FlightsCollection:
    def __init__(self, trajectories: list[Flights]):
        self._trajs = trajectories

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "FlightsCollection":
        return cls([Flights(icao, chunk.reset_index(drop=True))
                    for icao, chunk in df.groupby("icao24")])

    def __len__(self) -> int:
        return len(self._trajs)

    def __iter__(self) -> Iterator[Flights]:
        return iter(self._trajs)

    def filter_min_points(self, n: int) -> "FlightsCollection":
        return FlightsCollection([t for t in self._trajs if len(t) >= n])

    def filter_low_altitude(self, threshold_ft: float = 3000) -> "FlightsCollection":
        return FlightsCollection(
            [t for t in self._trajs
             if t.min_altitude is not None and t.min_altitude < threshold_ft]
        )

    def crop_to_bbox(self, minx, miny, maxx, maxy) -> "FlightsCollection":
        """Keep only the position points inside a lon/lat box (removes outliers)."""
        out = []
        for t in self._trajs:
            d = t.data
            d = d[d["longitude"].between(minx, maxx) & d["latitude"].between(miny, maxy)]
            if len(d):
                out.append(Flights(t.icao24, d.reset_index(drop=True)))
        return FlightsCollection(out)

    # ------------------------------------------------------------------ NEW --
    def min_altitudes(self) -> pd.Series:
        """Minimum altitude per flight, indexed by icao24 (NaN dropped)."""
        return pd.Series(
            {t.icao24: t.min_altitude for t in self._trajs}
        ).dropna().astype(float)

    def mean_vrates(self, altitude_max: Optional[float] = None) -> pd.Series:
        """Mean vertical rate per flight, indexed by icao24 (NaN dropped)."""
        return pd.Series(
            {t.icao24: t.mean_vrate(altitude_max=altitude_max) for t in self._trajs}
        ).dropna().astype(float)

    def by_phase(self, **phase_kwargs) -> dict[str, "FlightsCollection"]:
        """Return {'takeoff': ..., 'landing': ..., 'cruising': ...}."""
        buckets: dict[str, list[Flights]] = {"takeoff": [], "landing": [], "cruising": []}
        for t in self._trajs:
            buckets[t.phase(**phase_kwargs)].append(t)
        return {k: FlightsCollection(v) for k, v in buckets.items()}

    def takeoffs(self, **phase_kwargs) -> "FlightsCollection":
        return self.by_phase(**phase_kwargs)["takeoff"]

    def landings(self, **phase_kwargs) -> "FlightsCollection":
        return self.by_phase(**phase_kwargs)["landing"]
    # ------------------------------------------------------------------------

    def summary(self, **phase_kwargs) -> pd.DataFrame:
        """One row per flight, including its phase."""
        return pd.DataFrame(
            [{"icao24": t.icao24, "callsign": t.callsign, "count": len(t),
              "min": t.start, "max": t.end,
              "min_alt": t.min_altitude, "max_alt": t.max_altitude,
              "mean_vrate": t.mean_vrate(),
              "phase": t.phase(**phase_kwargs)}
             for t in self._trajs]
        ).set_index("icao24")

    def to_geodataframe(self, max_gap_s: Optional[float] = None,
                        epsg: int = 4326, **phase_kwargs) -> gpd.GeoDataFrame:
        rows = [
            {"icao24": t.icao24, "callsign": t.label,
             "min_alt": t.min_altitude, "max_alt": t.max_altitude,
             "mean_vrate": t.mean_vrate(),
             "phase": t.phase(**phase_kwargs),
             "geometry": line}
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

    # -------------------------------------------------------------- helpers --
    def _draw_basemap(self, ax) -> None:
        bm = self.basemap
        if len(bm.grande_couronne):
            bm.grande_couronne.plot(ax=ax, color="#D9D6D6",
                                    edgecolor="#6B6868", linewidth=0.3)
        bm.petite_couronne.plot(ax=ax, color="#A29F9F",
                                edgecolor="#0A0909", linewidth=0.5)
        bm.airports.plot(ax=ax, color="#A29F9F", edgecolor="#EE0808", linewidth=0.5)
        bm.airfields.plot(ax=ax, color="#EE0808", marker="^", markersize=30, zorder=5)
        for _, a in bm.airfields.iterrows():
            ax.annotate(a["name"], (a.geometry.x, a.geometry.y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", zorder=6)

    def _apply_zoom(self, ax, gdf: gpd.GeoDataFrame, zoom_on: str) -> None:
        if zoom_on == "petite_couronne":
            minx, miny, maxx, maxy = self.basemap.petite_couronne_bounds
        elif zoom_on == "trajectories" and len(gdf):
            minx, miny, maxx, maxy = gdf.total_bounds
        else:  # "basemap", or "trajectories" with nothing to draw
            minx, miny, maxx, maxy = self.basemap.bounds
        mx, my = (maxx - minx) * 0.03, (maxy - miny) * 0.03
        ax.set_xlim(minx - mx, maxx + mx)
        ax.set_ylim(miny - my, maxy + my)
        ax.set_axis_off()
    # ------------------------------------------------------------------------

    def diagnose(self, trajectories: FlightsCollection) -> None:
        g = trajectories.to_geodataframe(epsg=self.basemap.epsg)
        print("basemap      [minx miny maxx maxy]:", self.basemap.bounds.round(3))
        print("trajectories [minx miny maxx maxy]:", g.total_bounds.round(3))
        print("basemap CRS:", self.basemap.departments.crs)

    # ------------------------------------------------------------------ NEW --
    def plot_altitude_vrate_histograms(self, trajectories: FlightsCollection,
                                       bins: int = 40,
                                       altitude_max_for_vrate: Optional[float] = None,
                                       ax_size: tuple = (13, 5)):
        """Two side-by-side histograms to help pick the two thresholds:

        Left  : minimum altitude per flight -> gives the 'ground' altitude.
        Right : mean vertical rate per flight -> gives takeoff/landing thresholds.
        """
        fig, axes = plt.subplots(1, 2, figsize=ax_size)

        # ---- 1) minimum altitude per flight -------------------------------
        min_alts = trajectories.min_altitudes()
        axes[0].hist(min_alts, bins=bins, edgecolor="black", alpha=0.75, color="C0")
        axes[0].set_xlabel("Minimum altitude per flight (ft)")
        axes[0].set_ylabel("Number of flights")
        axes[0].set_title(f"Min altitude per flight   (n = {len(min_alts)})")
        axes[0].grid(True, alpha=0.3)
        if len(min_alts):
            for q, c, ls in [(0.25, "C3", "--"), (0.5, "k", "-"),
                             (0.75, "C3", "--")]:
                v = min_alts.quantile(q)
                axes[0].axvline(v, color=c, ls=ls, lw=1,
                                label=f"Q{int(q*100)} = {v:.0f} ft")
            axes[0].legend(fontsize=8, loc="upper right")

        # ---- 2) mean vrate per flight -------------------------------------
        mean_vr = trajectories.mean_vrates(altitude_max=altitude_max_for_vrate)
        axes[1].hist(mean_vr, bins=bins, edgecolor="black", alpha=0.75, color="C1")
        axes[1].set_xlabel("Mean vertical rate per flight (ft/min)")
        axes[1].set_ylabel("Number of flights")
        title = f"Mean vrate per flight   (n = {len(mean_vr)})"
        if altitude_max_for_vrate is not None:
            title += f"   |   below {altitude_max_for_vrate:.0f} ft"
        axes[1].set_title(title)
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(0, color="black", lw=1)
        if len(mean_vr):
            for q, c, ls in [(0.1, "C3", ":"), (0.25, "C3", "--"),
                             (0.5, "k", "-"),
                             (0.75, "C3", "--"), (0.9, "C3", ":")]:
                v = mean_vr.quantile(q)
                axes[1].axvline(v, color=c, ls=ls, lw=1,
                                label=f"Q{int(q*100)} = {v:.0f}")
            axes[1].legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        return fig, axes
    # ------------------------------------------------------------------------

    def plot(self, trajectories: FlightsCollection,
             max_gap_s: Optional[float] = None,
             zoom_on: str = "basemap",
             color_by_altitude: bool = True,
             color_column: str = "min_alt",
             labels: bool = False):
        fig, ax = plt.subplots(figsize=self.figsize)
        self._draw_basemap(ax)

        gdf = trajectories.to_geodataframe(max_gap_s, self.basemap.epsg)
        if len(gdf):
            if color_by_altitude:
                gdf.plot(ax=ax, column=color_column, cmap="viridis", linewidth=1,
                         legend=True,
                         legend_kwds={"label": f"{color_column} (ft)", "shrink": 0.6},
                         zorder=3)
            else:
                gdf.plot(ax=ax, color="#1f77b4", linewidth=1, zorder=3)

        if labels:
            for _, r in gdf.iterrows():
                x, y = r.geometry.coords[-1]
                ax.annotate(r["callsign"], (x, y), fontsize=6, zorder=4)

        self._apply_zoom(ax, gdf, zoom_on)
        return fig, ax

    # ------------------------------------------------------------------ NEW --
    def plot_phases(self, trajectories: FlightsCollection,
                    max_gap_s: Optional[float] = None,
                    zoom_on: str = "basemap",
                    takeoff_color: str = "#1a9850",   # green
                    landing_color: str = "#d73027",   # red
                    cruise_color:  str = "#4575b4",   # blue
                    show_cruise: bool = False,
                    phase_kwargs: Optional[dict] = None,
                    title: Optional[str] = None):
        """Same basemap as `plot`, with the three phases in distinct colours."""
        phase_kwargs = phase_kwargs or {}
        fig, ax = plt.subplots(figsize=self.figsize)
        self._draw_basemap(ax)

        parts = trajectories.by_phase(**phase_kwargs)
        takeoffs, landings, cruising = parts["takeoff"], parts["landing"], parts["cruising"]

        if show_cruise and len(cruising):
            g = cruising.to_geodataframe(max_gap_s, self.basemap.epsg)
            if len(g):
                g.plot(ax=ax, color=cruise_color, linewidth=0.7,
                       alpha=0.8, zorder=2)

        if len(takeoffs):
            g = takeoffs.to_geodataframe(max_gap_s, self.basemap.epsg)
            if len(g):
                g.plot(ax=ax, color=takeoff_color, linewidth=1.4, zorder=3)

        if len(landings):
            g = landings.to_geodataframe(max_gap_s, self.basemap.epsg)
            if len(g):
                g.plot(ax=ax, color=landing_color, linewidth=1.4, zorder=4)

        handles = [
            Line2D([0], [0], color=takeoff_color, lw=2.5,
                   label=f"Takeoff ({len(takeoffs)})"),
            Line2D([0], [0], color=landing_color, lw=2.5,
                   label=f"Landing ({len(landings)})"),
        ]
        if show_cruise and len(cruising):
            handles.append(Line2D([0], [0], color=cruise_color, lw=2.5,
                                  label=f"Cruising ({len(cruising)})"))
        ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

        if title:
            ax.set_title(title)

        all_gdf = trajectories.to_geodataframe(max_gap_s, self.basemap.epsg,
                                               **phase_kwargs)
        self._apply_zoom(ax, all_gdf, zoom_on)
        return fig, ax
    # ------------------------------------------------------------------------


# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    basemap = BaseMap.from_env()
    positions = AdsbLoader(os.environ["URL_Avions_MCD"]).load()

    trajs = FlightsCollection.from_dataframe(positions).filter_min_points(5)

    plotter = MapPlotter(basemap)

    minx, miny, maxx, maxy = basemap.bounds
    pad = 0.3
    local = trajs.crop_to_bbox(minx - pad, miny - pad, maxx + pad, maxy + pad)
    print(len(local), "aircraft near the Petite Couronne")

    # ------------------------------------------------------------------ #
    # 1) Diagnostic histograms to pick the two thresholds
    # ------------------------------------------------------------------ #
    fig_h, _ = plotter.plot_altitude_vrate_histograms(
        local,
        bins=40,
        altitude_max_for_vrate=None,   # set e.g. 5000 to focus on low-altitude
                                       # samples when computing mean vrate
    )

    # Print the quantiles so you can read them in the console too
    min_alts = local.min_altitudes()
    mean_vr  = local.mean_vrates()
    print("\n--- Min altitude per flight (ft) ---")
    print(min_alts.describe(percentiles=[.1, .25, .5, .75, .9]).round(1))
    print("\n--- Mean vrate per flight (ft/min) ---")
    print(mean_vr.describe(percentiles=[.1, .25, .5, .75, .9]).round(1))

    # ------------------------------------------------------------------ #
    # 2) Set the two thresholds from what you read on the histograms
    # ------------------------------------------------------------------ #
    TAKEOFF_MIN = 200.0    # ft/min  — flights with mean vrate >= this are climbing
    LANDING_MAX = -200.0   # ft/min  — flights with mean vrate <= this are descending
    ALT_MAX_FOR_VRATE = None   # or e.g. 5000 to restrict vrate to low altitude
    phase_kwargs = dict(takeoff_min=TAKEOFF_MIN,
                        landing_max=LANDING_MAX,
                        altitude_max=ALT_MAX_FOR_VRATE)

    # ------------------------------------------------------------------ #
    # 3) Classification report
    # ------------------------------------------------------------------ #
    parts = local.by_phase(**phase_kwargs)
    print(f"\n{len(parts['takeoff'])} takeoffs, "
          f"{len(parts['landing'])} landings, "
          f"{len(parts['cruising'])} cruising")
    print(local.summary(**phase_kwargs))

    # ------------------------------------------------------------------ #
    # 4) Maps
    # ------------------------------------------------------------------ #
    # Map 1: altitude-coloured overview
    plotter.plot(local, max_gap_s=120, zoom_on="basemap")

    # Map 2: takeoffs / landings / cruising in distinct colours
    plotter.plot_phases(local, max_gap_s=120, zoom_on="basemap",
                        phase_kwargs=phase_kwargs,
                        title="Takeoffs / landings / cruising over the Petite Couronne")

    plt.show()