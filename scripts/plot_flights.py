"""
Arrivals vs departures at Orly (ORY) and Roissy-CDG from jet1090 .jsonl files.

Reuses BaseMap and CprDecoder from plot_map.py (nothing in plot_map.py is changed).

Usage (from the project root):
    python scripts/plot_flights.py data/adsb_AKBAS_..._Arcueil.jsonl -o output/arcueil.png
    python scripts/plot_flights.py data/ensta_2026/adsb*.jsonl -o output/merged.png

Why this script exists
----------------------
A position message (latitude/longitude) never carries the callsign: the callsign
travels in a separate identification message. So `groupby(["icao24", "callsign"])`
puts every position row in a NaN callsign group, which pandas DROPS by default.
Here we group by icao24 only, and attach the callsign afterwards.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_map import BaseMap, CprDecoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Airport reference points (lat, lon)
AIRPORTS = {"ORY": (48.7233, 2.3794), "CDG": (49.0097, 2.5479)}

# Okabe-Ito blue / vermillion: distinguishable with every kind of colour blindness
COLORS = {"arrival": "#0072B2", "departure": "#D55E00", "other": "#C9C6C6"}
LABELS = {"arrival": "Arrivées", "departure": "Départs", "other": "Autres vols"}

KEEP = ["timestamp", "icao24", "callsign", "latitude", "longitude",
        "lat_cpr", "lon_cpr", "parity", "altitude"]


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def haversine_km(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def read_file(path: Path, decoder: CprDecoder, chunk: int = 200_000):
    """Stream one .jsonl (files reach 600 MB) and keep only the useful rows."""
    parts = []
    for ch in pd.read_json(path, lines=True, chunksize=chunk, convert_dates=False):
        for c in KEEP:
            if c not in ch:
                ch[c] = np.nan
        ch = ch[KEEP]
        parts.append(ch[ch[["latitude", "lat_cpr", "callsign"]].notna().any(axis=1)])
    raw = pd.concat(parts, ignore_index=True)

    # icao24 -> callsign
    cs = (raw.dropna(subset=["callsign"])
             .assign(callsign=lambda d: d.callsign.astype(str).str.strip().str.rstrip("_"))
             .drop_duplicates("icao24", keep="last")
             .set_index("icao24")["callsign"])

    # airborne positions only (they have an altitude); surface CPR is coded differently
    pos = raw[raw.altitude.notna() & (raw.latitude.notna() | raw.lat_cpr.notna())].copy()

    # Local CPR decoding for rows jet1090 left undecoded
    todo = pos.latitude.isna()
    sub = pos[todo]
    if len(sub):
        dec = [decoder.decode(a, b, p == "odd")
               for a, b, p in zip(sub.lat_cpr, sub.lon_cpr, sub.parity)]
        pos.loc[todo, ["latitude", "longitude"]] = np.array(dec)

    # Sanity check of the decoder on rows where jet1090 DID give a position
    ok = (~todo) & pos.lat_cpr.notna()
    if ok.sum():
        chk = pos[ok].head(5000)
        dec = np.array([decoder.decode(a, b, p == "odd")
                        for a, b, p in zip(chk.lat_cpr, chk.lon_cpr, chk.parity)])
        err = haversine_km(dec[:, 0], dec[:, 1], chk.latitude.values, chk.longitude.values)
        print(f"   decoder check on {len(chk)} rows: median error {np.median(err):.3f} km, "
              f"95th pct {np.percentile(err, 95):.3f} km")

    print(f"   {path.name}: {len(pos)} positions ({todo.sum()} decoded locally), "
          f"{pos.icao24.nunique()} aircraft")
    return pos[["timestamp", "icao24", "latitude", "longitude", "altitude"]], cs


def load_positions(paths: list[Path], ref=(48.80, 2.33)) -> pd.DataFrame:
    decoder = CprDecoder(*ref)
    all_pos, all_cs = [], []
    for p in paths:
        pos, cs = read_file(p, decoder)
        all_pos.append(pos)
        all_cs.append(cs)
    pos = pd.concat(all_pos, ignore_index=True)
    cs = pd.concat(all_cs).groupby(level=0).last()
    pos["callsign"] = pos.icao24.map(cs)
    # the same message heard by two receivers gives the same decoded point
    key = pd.DataFrame({"i": pos.icao24, "la": pos.latitude.round(5), "lo": pos.longitude.round(5)})
    return pos[~key.duplicated()]


# --------------------------------------------------------------------------- #
# 2. Flights = continuous pieces of track of one aircraft
# --------------------------------------------------------------------------- #
def split_flights(pos: pd.DataFrame, gap_s: float = 300) -> pd.DataFrame:
    pos = pos.sort_values(["icao24", "timestamp"]).reset_index(drop=True)
    new = (pos.icao24 != pos.icao24.shift()) | (pos.timestamp.diff() > gap_s)
    pos["flight_id"] = new.cumsum()
    return pos


def classify(pos: pd.DataFrame, radius_km: float = 25, low_ft: float = 5000,
             min_pts: int = 6, min_dalt_ft: float = 300) -> pd.DataFrame:
    """
    arrival   : passes within radius_km of ORY/CDG below low_ft AND is descending
    departure : same, but climbing
    other     : everything else (overflights, Le Bourget, Toussus...)
    """
    rows = []
    for fid, g in pos.groupby("flight_id"):
        if len(g) < min_pts:
            continue
        alt = g.altitude.to_numpy()
        k = max(2, len(g) // 4)
        d_alt = np.median(alt[-k:]) - np.median(alt[:k])  # end minus start
        kind, airport = "other", None
        near = []
        for name, (la, lo) in AIRPORTS.items():
            d = haversine_km(g.latitude.values, g.longitude.values, la, lo)
            i = d.argmin()
            if d[i] < radius_km and alt[i] < low_ft:
                near.append((d[i], name))
        if near and abs(d_alt) > min_dalt_ft:
            airport = min(near)[1]
            kind = "arrival" if d_alt < 0 else "departure"
        rows.append({"flight_id": fid, "icao24": g.icao24.iloc[0],
                     "callsign": g.callsign.iloc[0], "n": len(g),
                     "d_alt": d_alt, "kind": kind, "airport": airport})
    return pd.DataFrame(rows).set_index("flight_id")


# --------------------------------------------------------------------------- #
# 3. Plot
# --------------------------------------------------------------------------- #
def plot(basemap: BaseMap, pos: pd.DataFrame, info: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(9, 9))
    if len(basemap.grande_couronne):
        basemap.grande_couronne.plot(ax=ax, color="#EFEDED", edgecolor="#9A9696", linewidth=0.3)
    basemap.petite_couronne.plot(ax=ax, color="#DAD7D7", edgecolor="#6B6868", linewidth=0.5)
    basemap.airports.plot(ax=ax, color="#9A9696", edgecolor="#6B6868", linewidth=0.5)

    for kind, lw, z in (("other", 0.6, 2), ("arrival", 1.1, 3), ("departure", 1.1, 4)):
        ids = info.index[info.kind == kind]
        lines = [g[["longitude", "latitude"]].to_numpy()
                 for fid, g in pos[pos.flight_id.isin(ids)].groupby("flight_id")]
        ax.add_collection(LineCollection(lines, colors=COLORS[kind], linewidths=lw,
                                         alpha=0.8, zorder=z))

    for name, (la, lo) in AIRPORTS.items():
        ax.plot(lo, la, marker="o", ms=5, mfc="white", mec="black", zorder=6)
        ax.annotate(name, (lo, la), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", zorder=7)

    counts = info.kind.value_counts()
    handles = [Line2D([0], [0], color=COLORS[k], lw=2.5,
                      label=f"{LABELS[k]} ({counts.get(k, 0)})")
               for k in ("arrival", "departure", "other")]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9)

    minx, miny, maxx, maxy = basemap.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect(1 / np.cos(np.radians(48.8)))  # 1 deg lon < 1 deg lat at this latitude
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "output" / "flights.png")
    ap.add_argument("--idf", default=os.environ.get(
        "URL_IDF", str(ROOT / "geoflar-departements" / "geoflar-departements.shp")))
    ap.add_argument("--airports", default=os.environ.get(
        "URL_AIRPORTS", str(ROOT / "aeroports-dile-de-france" / "aeroports-dile-de-france0.shp")))
    ap.add_argument("--gap", type=float, default=300, help="max gap (s) inside one flight")
    ap.add_argument("--pickle", type=Path, help="cache of the merged positions")
    args = ap.parse_args()

    basemap = BaseMap(args.idf, args.airports)

    if args.pickle and args.pickle.exists():
        pos = pd.read_pickle(args.pickle)
    else:
        print("Reading", len(args.files), "file(s)")
        pos = load_positions(args.files)
        if args.pickle:
            pos.to_pickle(args.pickle)

    # drop CPR outliers: keep what is inside the Ile-de-France box (+ margin)
    minx, miny, maxx, maxy = basemap.bounds
    pad = 0.3
    pos = pos[pos.longitude.between(minx - pad, maxx + pad)
              & pos.latitude.between(miny - pad, maxy + pad)
              & pos.altitude.between(-1000, 60000)]

    pos = split_flights(pos, args.gap)
    info = classify(pos)
    print(info.kind.value_counts().to_string())
    print(info[info.kind != "other"].groupby(["kind", "airport"]).size().to_string())

    n_sources = len(args.files)
    title = (f"Trajectoires ADS-B : arrivées et départs ORY / CDG "
             f"({n_sources} source{'s' if n_sources > 1 else ''})")
    fig, _ = plot(basemap, pos[pos.flight_id.isin(info.index)], info, title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
