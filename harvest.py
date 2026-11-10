"""
Harvest live driving travel times between fixed point pairs (Mapbox Directions API).

Each run queries every pair once and appends one row per pair to data/travel_times.csv.
Run it on a schedule (GitHub Actions workflow included) every 10 minutes, noon to 6pm ET.

Setup:
  1. Make a free Mapbox account and copy your public access token.
  2. Set it as an environment variable:  export MAPBOX_TOKEN=pk.xxxxx
  3. Test once by hand:                  python harvest.py --force
"""

import csv
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

TOKEN = os.environ.get("MAPBOX_TOKEN")
OUT_FILE = "data/travel_times.csv"
ET = ZoneInfo("America/New_York")

# Collection window (local Boston time). Runs outside it exit without querying.
START_HOUR = 12   # noon
END_HOUR = 18     # 6pm

# (name, origin (lat, lon), destination (lat, lon))
# Coordinates are approximate square centers. Check each on a map and adjust.
# Swap one or two pairs onto a current construction zone once you find one.
PAIRS = [
    ("harvard_to_porter",   (42.3734, -71.1189), (42.3884, -71.1191)),
    ("porter_to_harvard",   (42.3884, -71.1191), (42.3734, -71.1189)),
    ("harvard_to_central",  (42.3734, -71.1189), (42.3654, -71.1037)),
    ("central_to_kendall",  (42.3654, -71.1037), (42.3625, -71.0862)),
    ("harvard_to_kendall",  (42.3734, -71.1189), (42.3625, -71.0862)),
    ("porter_to_davis",     (42.3884, -71.1191), (42.3967, -71.1223)),
    ("porter_to_alewife",   (42.3884, -71.1191), (42.3954, -71.1425)),
    ("harvard_to_inman",    (42.3734, -71.1189), (42.3740, -71.1007)),
    ("harvard_to_common",   (42.3734, -71.1189), (42.3550, -71.0656)),
    ("kendall_to_common",   (42.3625, -71.0862), (42.3550, -71.0656)),
]

FIELDS = [
    "timestamp_et", "weekday", "pair",
    "duration_s", "duration_typical_s", "distance_m", "route_summary",
]


def get_route(origin, dest):
    """Return one traffic-aware route between two (lat, lon) points."""
    # Mapbox wants lon,lat order
    coords = f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/{coords}"
    params = {"access_token": TOKEN, "overview": "false", "alternatives": "false"}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    route = resp.json()["routes"][0]
    return {
        "duration_s": round(route["duration"], 1),
        # typical (historical) time for this departure time, if Mapbox returns it
        "duration_typical_s": route.get("duration_typical", ""),
        "distance_m": round(route["distance"], 1),
        # main roads used, so you can see when drivers get rerouted
        "route_summary": route["legs"][0].get("summary", ""),
    }


def main():
    if not TOKEN:
        sys.exit("MAPBOX_TOKEN is not set.")

    now = datetime.now(ET)
    force = "--force" in sys.argv
    if not force and not (START_HOUR <= now.hour < END_HOUR):
        print(f"{now:%H:%M} ET is outside {START_HOUR}:00-{END_HOUR}:00. Skipping.")
        return

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    new_file = not os.path.exists(OUT_FILE)

    rows = []
    for name, origin, dest in PAIRS:
        try:
            result = get_route(origin, dest)
        except Exception as err:  # keep going if one pair fails
            print(f"{name}: failed ({err})")
            continue
        row = {
            "timestamp_et": now.strftime("%Y-%m-%d %H:%M"),
            "weekday": now.strftime("%a"),
            "pair": name,
            **result,
        }
        rows.append(row)
        print(f"{name}: {row['duration_s'] / 60:.1f} min via {row['route_summary']}")

    with open(OUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
