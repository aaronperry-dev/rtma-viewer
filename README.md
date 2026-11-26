# New England RTMA / URMA viewer

Hourly maps of **34 surface variables** from NOAA's **RTMA** and **URMA** (2.5 km), each as the current field plus
**1 / 3 / 6 / 12 / 24-hour change** where a change is meaningful, published as a static dashboard on GitHub Pages:

* **Basics:** temperature, dew point, apparent temperature, wind speed / gust / **direction**, **surface pressure**,
  **visibility**, **cloud ceiling**, **cloud cover**, terrain height.
* **Moisture and pressure:** **wet bulb**, **relative humidity**, **dew point depression**, **RH with respect to ice**,
  **mixing ratio**, **mean sea-level pressure (with isobars)**, **pressure-gradient strength**.
* **Winter hazards and cold-air diagnostics:** **sea-level-adjusted temperature**, **flight category**, **frostbite time**, and
  **past-24-hour** maximum / minimum / range, **hours at or below 32 F**, **freeze-thaw crossings**, **minimum wind chill**
  and **maximum gust**.
* **Surface kinematics:** **temperature and dew point advection**, **convergence**, **vorticity**, **frontogenesis** and
  **terrain-forced (upslope) lift** from the 10 m wind.

Everything - fetching, decoding, plotting, site generation - is Python; the page itself is plain HTML + a little JS.

```
rtma_viewer/
  s3.py          anonymous S3 access, .idx parsing, one HTTP byte-range request per GRIB message
  grib.py        eccodes decode straight from memory, New England window of the CONUS grid
  fields.py      fetch -> decode -> window -> compact cache (float16 / packed uint16 / static float32; each hour only the newest cycle is downloaded)
  colors.py      fixed-level colour ramps (NWS wind / wind chill / heat index legends; WPC-style ramp for changes)
  derived.py     wind chill, heat index, apparent temperature, wet bulb, MSLP, RH (liquid / ice), mixing ratio, sea-level temperature,
                 flight category, frostbite zones, 24-hour max / min / hours-below-freezing / crossings, calm-masked wind direction
  kinematics.py  grid geometry + advection, convergence, vorticity, frontogenesis, terrain lift, pressure gradient
  stations.py    New England METAR stations (data/stations.csv) + label de-cluttering
  boundaries.py  Natural Earth 1:10m states/counties/coast/lakes, clipped + pre-projected once
  render.py      cartopy/matplotlib map panels -> palette-quantised, oxipng-recompressed PNGs
  build.py       orchestrator: writes public/{index.html,data.json,img/*.png}
  testing.py     fake S3 serving real GRIB2 + .idx + Range (for --demo and tests)
site/index.html  dashboard (product / variable / time-step controls, compare grid, table view)
.github/workflows/update.yml   hourly build + deploy
```

## Publish it (one-time)

1. Push this folder to a new GitHub repository (public repos get free Actions + Pages).
2. **Settings -> Pages -> Build and deployment -> Source: GitHub Actions.**
3. **Actions -> "Update RTMA/URMA viewer" -> Run workflow.** After that it runs at :47 every hour
   and the site is at `https://<user>.github.io/<repo>/`.

## What is (not) stored in git

Nothing generated. The workflow builds into `public/` and deploys it with the Pages artifact
actions, so **no PNGs, GRIBs or data files are ever committed** - the repo stays source-only and
does not grow. The only persisted state is a small Actions cache (`.cache/fields`, compact float16 / packed arrays
of the last ~28 h, ~65 MB per product, ~130 MB in all) and the Natural Earth shapefiles; the workflow keeps the newest two
field caches and deletes the rest.

## Data used

* Only the needed GRIB messages are requested: the `.idx` gives each message's start byte, and
  `get_object(Range="bytes=start-end")` downloads just that message instead of the whole file. The messages used are
  `TMP` `DPT` `WIND` `GUST` `WDIR` `UGRD` `VGRD` (2 m / 10 m), `PRES` and `HGT` (surface), `VIS`, `CEIL` and `TCDC`; the file's
  `SPFH` is not needed (WIND falls back to hypot(UGRD, VGRD) if absent). `UGRD` / `VGRD` feed only the kinematic maps.
* Variables are only loaded for the hours their maps need (`hours_needed()` in `config.py`, which walks the derivation graph):
  dew point and pressure for 6 cycles (the change maps), **temperature, wind and gust for all 25 hourly cycles** of the past day
  (the 24-hour summaries), and wind direction, visibility, ceiling, cloud cover and the wind components only for the newest cycle;
  terrain (static) once per product.
* A normal run downloads the newest cycle only: about **70 MB per product** (~140 MB per hourly run, up from ~116 MB), because
  every earlier hour was once the newest and is already in the cache. A cold start (or cache eviction) fetches 25 cycles of
  temperature, wind and gust: **~560 MB per product** (measured; about a minute per product on a fast link). The build log prints
  real bytes and request counts for each product. If a few hourly analyses are missing the 24-hour summaries still draw (at least
  19 of 25 are needed; hours below freezing is rescaled to the full period).
* Cache precision: temperature-like fields are float16 (<= 0.06 units), but float16 would quantise ~1000 mb pressure to 0.5-1 mb,
  so pressure is cached as packed uint16 at 0.01 mb (same 2 bytes per cell), and the wind components (which are differentiated) at
  0.002 m/s. Variables that only need the newest cycle are pruned to one file. The field cache is about 65 MB per product.
* RTMA is published ~35-45 min after the hour; URMA a few hours later. The build looks back up to
  6 h (RTMA) / 12 h (URMA) for the newest published cycle, so the page always shows the latest
  available analysis and states its valid time and lag.

## PNG size

Maps are drawn at 1200x876, then converted to an indexed palette made of the **exact** colour-bin
colours plus a median-cut palette for text/line anti-aliasing (no dithering), then recompressed
losslessly with `oxipng` when available. Expect roughly 120-180 KB per map.
(`--colors` and `--dpi` trade size for fidelity.)

## Map style

Landscape page with a thin-framed map: header line `RTMA: 2m Temperature (°F)` (left) and
`Valid: 17:00z Thu Nov 5 2026` (right), a wide segmented colour bar bottom-left, thick state lines over
thin county lines (Natural Earth 1:10m), and the footer credit `aaronperry-dev.github.io (Aaron Perry)`
(`BRANDING` in `render.py`).

**Values are labelled at METAR stations**, not on a grid: the analysis cell nearest each airport is
sampled and printed in bold with a white halo (values only; set `SHOW_STATION_IDS = True` in `render.py`
to add the ICAO id underneath).
`rtma_viewer/data/stations.csv` holds 79 New England airports plus **KMWN** (Mount Washington summit,
always labelled). Coordinates come from [OurAirports](https://ourairports.com/data/) via
`scripts/make_stations.py`, which also holds the curated ICAO list - edit it and re-run to add or drop
sites. When the map is crowded, higher-priority sites win and neighbours closer than 40 km are skipped.
Station values are also written to `data.json`. (RTMA's grid terrain is smoothed, so a summit site such
as KMWN reads the analysis at the grid's terrain height, not at 1917 m.)

## Colour ramps

* Fixed physical levels (same colours for the same value in every frame/product).
* Temperature: the classic -40..120 F operational ramp (magenta -> purple -> blue | green at 32 F ->
  yellow -> orange -> red -> pink), 1 F bins. Dew point: a sequential ramp, monotone in lightness.
* **Wind speed and wind gust**: the NWS "Maximum Wind Gust" legend (sampled from an NWS Gray/Portland map) -
  17 whole-number bins (0-5, 5-10 ... 50-60, 60-70, 70-80, 80-100, 100-120, 120-140, 140-150 mph) with an equal-width legend.
* **Apparent temperature**: the NWS "24-Hour Minimum Wind Chill Index Temperature" legend (< -60 ... 35+, 5 F bins)
  joined to the NWS "24-Hour Maximum Heat Index Temperature" legend (75 ... 120+, 5 F bins).  Neither legend covers
  35-75 F (wind chill is one flat colour from 35 F up; heat index is a single grey below 75 F), so that span is a
  smooth OKLab bridge from the wind-chill teal to the first heat-index yellow.
* **New variables**: wet bulb reuses the temperature ramp; relative humidity, dew point depression, cloud cover, surface pressure and terrain
  have sequential ramps; MSLP is diverging about ~1013 mb (2 mb bins); wind direction is a cyclic 16-point compass; visibility and
  ceiling use aviation-style category breaks (every bin edge is a flight-category boundary). A value exactly on the top level (RH or cloud
  cover of 100 %, direction 360) takes the top colour rather than becoming transparent.
* Changes (every variable that has them, all intervals): the WPC-style change ramp - white for -1..+1, blues for cooling,
  peach -> orange -> red -> maroon -> magenta for warming - with the original whole-number levels (1, 3, 5, 8, 12,
  16, 20, 25, 30, 35, 40, 50) and an equal-width legend. The levels are the same for 1/3/6/12/24 h. Pressure and MSLP use the same
  25 colours on finer levels (+-0.5, 1, 2, 3, 4, 6, 9, 12, 16, 20, 24, 30 mb) because pressure changes are an order of magnitude smaller.
* Edit `rtma_viewer/colors.py` to change anchors, ranges or bin widths.

## Variables

| Variable | Source | Change maps | Notes |
|---|---|---|---|
| Temperature, dew point | `TMP`, `DPT` 2 m | yes | |
| Apparent temperature | derived: T, Td, wind | yes | see below |
| **Wet bulb** | derived: T, Td, surface pressure | yes | thermodynamic wet bulb (liquid water); same legend as temperature, so the 32 F break lines up |
| **Relative humidity** | derived: T, Td | yes | with respect to liquid water, 0-100 % |
| **Dew point depression** | derived: T - Td | yes | near 0 = saturated (fog, low cloud, rime) |
| **Sea-level temperature** | derived: T, terrain | no | T + 3.57 F per 1,000 ft (6.5 K/km); temperature legend; removes elevation so cold pools / air-mass edges show |
| **RH w.r.t. ice** | derived: T, Td | no | over ice below 0 C, water above; **not clipped at 100** (saturated-over-water air reads ~110 % at 15 F); violet above 100 % |
| **Mixing ratio** | derived: Td, surface pressure | no | g/kg; follows the air mass's moisture independent of temperature |
| **24 h max / min temperature, range** | derived: 25 hourly T | no | the 24 hours ending at the valid time (hourly snapshots, not calendar days) |
| **Hours <= 32 F** | derived: 25 hourly T | no | linear interpolation between hourly analyses |
| **Freeze-thaw crossings** | derived: 25 hourly T | no | times T crossed 32 F in either direction (2+ is a freeze-thaw cycle) |
| **24 h min wind chill** | derived: 25 hourly T, wind | no | NWS "24-Hour Minimum Wind Chill" legend; air temperature where wind chill is undefined |
| **24 h max gust** | derived: 25 hourly `GUST` | no | NWS wind legend |
| Wind speed, gust | `WIND`, `GUST` 10 m | yes | NWS wind legend |
| **Wind direction** | `WDIR` masked where wind < 2 mph | no | 16-point compass legend; calm air (RTMA encodes it as 0) is left white; average is a vector mean |
| **Frostbite time** | derived: T, wind | no | none / 30 / 10 / 5 min from the NWS wind chill chart (round mid-chart values, see below); area shares instead of low / high |
| **Surface pressure** | `PRES` | yes | station pressure at the analysis terrain height (mostly a map of elevation); change maps use finer levels (0.5 mb centre, 30 mb ends) |
| **MSLP** | derived: PRES, HGT, T, Td | yes | isobars every 2 mb (labelled every 4) from a lightly smoothed field; summit stations above 1,000 m are not labelled |
| **Pressure gradient** | derived: MSLP, terrain | no | mb per 100 km of MSLP smoothed over ~60 km; blank above 1,000 m; ~17 mph geostrophic wind per mb / 100 km |
| **Visibility** | `VIS` | no | statute miles; the analysis tops out near 10 mi |
| **Cloud ceiling** | `CEIL` | no | feet AGL, aviation breaks (500 / 1,000 / 3,000 ft); 12,000 ft and up (or none) is `CLR` |
| **Flight category** | derived: visibility, ceiling | no | LIFR / IFR / MVFR / VFR (the worse of the two), shown as area shares |
| **Cloud cover** | `TCDC` | no | total sky cover, % |
| **Temperature / dew point advection** | derived: sea-level T / Td, `UGRD` `VGRD` | no | F per hour, warm / moist positive; see "Surface kinematics" |
| **Convergence, vorticity** | derived: `UGRD` `VGRD` | no | 1e-5 s-1; convergence positive = air piling up, vorticity positive = cyclonic |
| **Frontogenesis** | derived: sea-level T, `UGRD` `VGRD` | no | Petterssen 2-D kinematic, F per 100 km per 3 h |
| **Upslope lift** | derived: `UGRD` `VGRD`, terrain | no | V . grad(z), cm/s; positive = blowing uphill |
| **Terrain height** | `HGT` (static) | no | the analysis grid's smoothed terrain, in feet |

Current-only variables (no change maps) have 1 / 3 / 6 / 12 / 24-hour buttons greyed out; switching to one from a change map falls back
to the current field. Every variable has a short "how to read this" note on the page (from `Variable.note` in `config.py`).

### Mean sea-level pressure (derived)

RTMA / URMA publish surface pressure and terrain height but **no sea-level pressure**, so MSLP is reduced with the hypsometric equation,
`MSLP = P * exp(g z / (Rd * Tv_mean))`, `Tv_mean = Tv_2m + 0.5 * 6.5 K/km * z`, using the virtual temperature (so humidity is
accounted for) and the analysis terrain height `z`. It reproduces the ICAO standard atmosphere, and was checked against **observed METAR
sea-level pressure at seven New England airports on the 2026-11-26 02Z analysis: bias -0.06 mb, rms 0.37 mb, worst 0.6 mb**
(that case is embedded in `tests/test_derived.py`). The reduction is sensitive to terrain height (~1.2 mb per 10 m near sea level) and is
less certain over high terrain and in cold pools, which is why isobars come from a smoothed field and summit stations are not labelled.

### Wet-bulb temperature (derived)

The temperature the air reaches when water evaporates into it at constant pressure until saturated (isobaric adiabatic saturation):
the root of `(cp_d + r cp_v)(T - Tw) = Lv(Tw) (r_s(Tw) - r)`, solved by vectorised bisection between the dew point and the temperature.
It uses the real surface pressure, so high ground reads correctly. Against MetPy's wet-bulb temperature on 500 random states
(T -25..30 C, RH 20-100 %, 820-1040 mb): 0.08 C rms overall, **0.02 C rms and 0.06 C worst below 2 C**. Rule of thumb for
precipitation type: precipitation falling into a surface wet bulb at or below 32 F tends to stay frozen, well above 34-35 F favours rain,
and in between is marginal or mixed (the map is not a precipitation forecast).

### Past-24-hour products and winter hazards (derived)

These need no extra download: they are computed from the hourly analyses the cache already holds (every hour was once the newest).
`hours_at_or_below` interpolates linearly between hourly analyses (a ramp from 20 F to 44 F spends exactly 12 h at or below 32 F;
tested), and `threshold_crossings` counts sign changes of `T - 32`. Maxima and minima are over hourly snapshots, so a brief extreme
between two analyses is missed, and the window ends at the valid time rather than at a calendar day.

**Flight category** follows the FAA / NWS definitions on the analysed visibility and ceiling (VFR: ceiling > 3,000 ft *and* visibility
> 5 mi; MVFR 1,000-3,000 ft and/or 3-5 mi; IFR 500-999 ft and/or 1 to < 3 mi; LIFR < 500 ft and/or < 1 mi). The analysed visibility
and ceiling are coarse, so it is guidance, not a METAR.

**Frostbite time** uses the NWS wind chill formula and approximate zone edges from the NWS chart: about 30 minutes at a wind chill of
-18 F or colder, 10 minutes at -33 F, 5 minutes at -48 F. The chart's shading is not a single wind-chill contour (its edges move a few
degrees with wind speed), so treat these as round mid-chart values. In light wind (3 mph or less, where wind chill is undefined) the
air temperature itself is compared.

**Sea-level temperature** is `T + 6.5 K/km * z` (3.57 F per 1,000 ft); it is the thermal field the advection and frontogenesis maps
use, because otherwise wind blowing along a slope would register as warm or cold advection.

**RH with respect to ice** uses Alduchov-Eskridge saturation pressures over ice below 0 C (1.032 hPa at -20 C and 0.128 hPa at -40 C,
matching standard tables) and over liquid water above. The published dew point is with respect to liquid water, so saturated air
reads above 100 % below freezing (121 % at -20 C): that is supersaturation over ice, not an error.

### Surface kinematics (derived)

Advection, convergence, vorticity, frontogenesis and terrain lift are computed in `rtma_viewer/kinematics.py` from the 10 m wind
components. Things worth knowing:

* **The components are grid-relative, not east / north.** RTMA / URMA set GRIB2 `uvRelativeToGrid = 1`; on this Lambert-conformal grid
  the grid's x axis is rotated -n (lon - lon0) from east (n = sin 25 deg), ~10.6 deg over New England. The diagnostics therefore
  difference the components directly along the grid axes, using the true ground distance between cells (from latitude / longitude,
  so the map-scale factor is included). `tests/test_kinematics.py` pins the convention against 60 real RTMA cells (direction from
  `UGRD` / `VGRD` vs the published `WDIR` differ by exactly the convergence angle, within 0.1 deg), so a future change in NOAA's
  convention fails the tests instead of silently rotating every map.
* **Smoothing first.** Derivatives of a 2.5 km wind analysis are mostly noise, so the wind and the scalar are smoothed with a
  Gaussian (sigma 3 cells, ~7.6 km; 4 for frontogenesis) before centred differencing. Features narrower than ~20 km are
  not resolved, and coastlines / lake shores light up because land and water really differ.
* Formulas: advection `-V . grad(s)`; convergence `-(du/dx + dv/dy)`; vorticity `dv/dx - du/dy`; Petterssen frontogenesis
  `-(1/|grad T|) [Tx^2 ux + Ty^2 vy + Tx Ty (vx + uy)]`; terrain lift `V . grad(z)`. Divergence, vorticity, advection and
  frontogenesis are checked against **MetPy** on random smooth fields (agreement to 0.1-2 %), and against analytic cases (solid-body
  rotation, uniform expansion, linear gradients, sloping planes).
* These are 10 m, surface-only, horizontal-kinematic diagnostics: no vertical motion, no diabatic heating, nothing above the surface
  layer. Use them to see *where* boundaries and lift are, not for exact values. Frontogenesis and terrain lift are the most
  qualitative; terrain lift ignores stability, so blocked low-level flow does not actually climb the terrain.
* **Pressure gradient.** The 2.5 km MSLP carries terrain-scale noise (10 m of terrain error is ~1 mb), so its raw gradient is mostly
  noise: on a live day with a broad ~0.7 mb / 100 km gradient, sigma = 5 cells read 1.6, 12 read 1.1 and 25 (used, ~60 km) ~0.85.
  Narrow features such as a coastal front are therefore smoothed away; the map is for the broad gradient (wind potential).

## Apparent temperature (derived product)

Computed on the analysis grid by `rtma_viewer/derived.py` from RTMA/URMA 2 m temperature, 2 m dew point and 10 m wind
speed - no extra GRIB download beyond the dew point message. It follows the NWS NDFD definition:

| Where | Apparent temperature |
|---|---|
| T <= 50 F and wind > 3 mph | NWS **wind chill**: `35.74 + 0.6215 T - 35.75 V^0.16 + 0.4275 T V^0.16` (T in F, V in mph) |
| T > 80 F | **Heat index**, WPC algorithm: Steadman's simple formula averaged with T; where that is >= 80 F the Rothfusz regression, with the low-humidity (RH < 13 %, 80-112 F) and high-humidity (RH > 85 %, 80-87 F) adjustments |
| otherwise | ambient air temperature |

The heat index needs relative humidity, which is not an RTMA field; it is derived from temperature and dew point
(Magnus formula, clipped to 0-100 %). So the product uses temperature, wind **and dew point**.  Changes are computed
on the apparent-temperature fields, so a cell that crosses 50 F or 80 F switches regime between the two times.
Sources: [NWS wind chill](https://www.weather.gov/safety/cold-wind-chill-chart),
[WPC heat index equation](https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml), and the NDFD apparent
temperature definition (see `derived.py`).

## Run locally

```bash
pip install -r requirements.txt
python -m rtma_viewer.build --demo          # offline, synthetic data (fake S3 serving real GRIB2)
python -m rtma_viewer.build                 # live NOAA data
python -m http.server -d public 8000        # http://localhost:8000
pytest                                      # idx parsing, ramps, derived-field physics (real METAR / real-wind and MetPy reference cases), kinematics, byte-range fetch + decode + cache
```

## Tuning

| Want | Change |
|---|---|
| Different region / frame | `MAP_CENTER`, `HALF_H_M` in `render.py` and `DISPLAY_BBOX` in `config.py` (checked at import) |
| Stations / label density | `scripts/make_stations.py`, `MIN_SEP_M` in `render.py` |
| Another variable | add a `Variable` in `config.py` (+ a style in `colors.py`). Useful flags: `derive` / `requires` (computed from other variables), `changes=False` (current field only), `series=N` (summarise the last N hours: `derive` gets stacks), `needs_geom=True` (kinematics: `derive(..., geom=)`), `static=True` (terrain-like), `hidden=True` (input only, no map), `pack=(offset, scale)` (cache precision), `diverging=True` (signed current field), `categories=(...)` (integer classes), `group`, `note` |
| Time steps | `INTERVALS` in `config.py` |
| Schedule | cron in `update.yml` |
