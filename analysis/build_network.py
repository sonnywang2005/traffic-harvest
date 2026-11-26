"""
Step 5a: download the Cambridge/Boston main-road network for the road add/remove analysis.

Run once on your laptop from the repo root (needs internet; OpenStreetMap):
    pip3 install osmnx scikit-learn
    python3 analysis/build_network.py

Saves to data/network/:
    cambridge_drive.graphml   full network (open again with osmnx.load_graphml)
    edges.csv                 one row per road segment: from, to, length, lanes, speed, type, name
    nodes.csv                 one row per intersection: id, lat, lon
    pair_nodes.csv            the network intersection closest to each of the 10 harvest points,
                              plus a check that the network route length matches Mapbox's
"""

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harvest import PAIRS  # the 10 (name, origin, destination) pairs  # noqa: E402

OUT = ROOT / "data" / "network"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# 1. Download main roads in a box covering all 10 pairs
#    (Alewife/Davis in the northwest to Boston Common in the southeast)
# ---------------------------------------------------------------
CENTER = (42.376, -71.105)      # roughly Inman Square
HALF_WIDTH_M = 4500             # box is 9 km x 9 km
ROAD_TYPES = ('["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|'
              'secondary|secondary_link|tertiary|tertiary_link"]')

# Public OpenStreetMap servers get busy; try several, waiting longer between rounds.
SERVERS = ["https://overpass-api.de/api",
           "https://overpass.kumi.systems/api",
           "https://overpass.private.coffee/api",
           "https://maps.mail.ru/osm/tools/overpass/api"]
ox.settings.requests_timeout = 300

G = None
for attempt in range(3):
    for url in SERVERS:
        ox.settings.overpass_url = url
        print(f"Downloading roads from {url} (about 1 minute)...")
        try:
            G = ox.graph_from_point(CENTER, dist=HALF_WIDTH_M, network_type="drive",
                                    custom_filter=ROAD_TYPES)
            break
        except Exception as err:          # server busy or timed out: try the next one
            print(f"  failed: {str(err).splitlines()[0][:120]}")
    if G is not None:
        break
    wait = 30 * (attempt + 1)
    print(f"All servers busy. Waiting {wait} s, then trying again...")
    time.sleep(wait)
if G is None:
    sys.exit("Could not reach any OpenStreetMap server. Try again in an hour.")

# keep the largest piece where every intersection can reach every other one
largest = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(largest).copy()
print(f"Network: {G.number_of_nodes()} intersections, {G.number_of_edges()} road segments")

# ---------------------------------------------------------------
# 2. Clean the attributes the model needs
# ---------------------------------------------------------------
DEFAULT_MPH = {"motorway": 55, "trunk": 40, "primary": 30, "secondary": 25, "tertiary": 25}
DEFAULT_LANES = {"motorway": 3, "trunk": 2, "primary": 2, "secondary": 1, "tertiary": 1}


def first(value):
    """OSM sometimes stores a list (e.g. two road types); keep the first entry."""
    return value[0] if isinstance(value, list) else value


def road_class(hw):
    hw = str(first(hw)).replace("_link", "")
    return hw if hw in DEFAULT_MPH else "tertiary"


def parse_mph(value, hw):
    try:
        return float(str(first(value)).split()[0])
    except (TypeError, ValueError, IndexError):
        return float(DEFAULT_MPH[road_class(hw)])


def parse_lanes(value, hw, oneway):
    try:
        lanes = float(str(first(value)).split(";")[0])
    except (TypeError, ValueError):
        return float(DEFAULT_LANES[road_class(hw)])
    # OSM "lanes" counts both directions on two-way roads; we need lanes per direction
    return lanes if oneway else max(1.0, lanes / 2)


rows = []
for u, v, k, d in G.edges(keys=True, data=True):
    hw = d.get("highway")
    oneway = bool(first(d.get("oneway", False)))
    rows.append({
        "u": u, "v": v, "key": k,
        "length_m": round(float(d["length"]), 1),
        "lanes": parse_lanes(d.get("lanes"), hw, oneway),
        "speed_mph": parse_mph(d.get("maxspeed"), hw),
        "road_class": road_class(hw),
        "oneway": oneway,
        "name": str(first(d.get("name", ""))),
        "lanes_from_osm": d.get("lanes") is not None,
        "speed_from_osm": d.get("maxspeed") is not None,
    })
edges = pd.DataFrame(rows)
edges.to_csv(OUT / "edges.csv", index=False)

nodes = pd.DataFrame([{"node": n, "lat": d["y"], "lon": d["x"]} for n, d in G.nodes(data=True)])
nodes.to_csv(OUT / "nodes.csv", index=False)
ox.save_graphml(G, OUT / "cambridge_drive.graphml")

print(f"Lanes from OSM on {edges.lanes_from_osm.mean():.0%} of segments, "
      f"speed limits on {edges.speed_from_osm.mean():.0%} (the rest use road-type defaults)")

# ---------------------------------------------------------------
# 3. Snap the 10 harvest points to the network and sanity-check distances
# ---------------------------------------------------------------
tt = ROOT / "data" / "travel_times.csv"
mapbox_dist = pd.read_csv(tt).groupby("pair").distance_m.median() if tt.exists() else pd.Series(dtype=float)

pair_rows = []
for name, origin, dest in PAIRS:
    o = ox.distance.nearest_nodes(G, X=origin[1], Y=origin[0])
    d = ox.distance.nearest_nodes(G, X=dest[1], Y=dest[0])
    o_gap = ox.distance.great_circle(origin[0], origin[1], G.nodes[o]["y"], G.nodes[o]["x"])
    d_gap = ox.distance.great_circle(dest[0], dest[1], G.nodes[d]["y"], G.nodes[d]["x"])
    try:
        net_len = nx.shortest_path_length(G, o, d, weight="length")
    except nx.NetworkXNoPath:
        net_len = np.nan
    mb = mapbox_dist.get(name, np.nan)
    pair_rows.append({
        "pair": name, "origin_node": o, "dest_node": d,
        "origin_snap_m": round(o_gap), "dest_snap_m": round(d_gap),
        "network_shortest_m": round(net_len) if net_len == net_len else None,
        "mapbox_median_m": round(mb) if mb == mb else None,
        "ratio": round(net_len / mb, 2) if mb == mb and net_len == net_len else None,
    })
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(OUT / "pair_nodes.csv", index=False)

print("\nHarvest points on the network (snap = distance from point to nearest intersection):")
print(pairs.to_string(index=False))
print("\nCheck: snaps should be under ~300 m, and 'ratio' (network route / Mapbox route)")
print("should be roughly 0.8 to 1.3. Larger gaps mean a point sits on a side street")
print("that the main-road network leaves out; that is fine, but note it.")
print(f"\nSaved to {OUT}")
