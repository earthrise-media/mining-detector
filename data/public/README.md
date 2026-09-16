# Publicly served files

These paths are public URLs, linked straight from the Amazon Mining Watch website.

- [`mined_areas_by_jurisdiction.csv`](#mined_areas_by_jurisdictioncsv) — mined area per jurisdiction, per period, 2018 onward
- [`illegality_analysis_by_jurisdiction.csv`](#illegality_analysis_by_jurisdictioncsv) — that area split by illegality risk, latest period only

Both are written by
[`scripts/boundaries/collate_jurisdiction_areas.py`](../../scripts/boundaries/collate_jurisdiction_areas.py).

## `mined_areas_by_jurisdiction.csv`

Mined area within each Amazon jurisdiction, by period, from 2018 onward.

**`intersected_area_ha_cumulative` is the column most people want**: total
hectares of mining detected inside that jurisdiction from 2018 through the given
period. `intersected_area_ha` gives the increment added during that one
period.

Areas come from intersecting the mining scar raster masks with the jurisdiction
boundaries in [`data/boundaries/`](../boundaries). The masks themselves are
published separately:
<https://data.source.coop/earthgenome/amazon-mining-watch/amazon_basin_mining_scar_masks.tif>

| column | meaning |
| --- | --- |
| `date_published` | which data publish this file was built from. Areas are restated when we reprocess, so quote this alongside any figure |
| `id`, `name`, `country`, `country_code` | the jurisdiction. `id` is stable across publishes; `AMAZ` is the basin-wide total |
| `type` | `national_admin`, `subnational_admin`, `indigenous_territories`, or `protected_areas` |
| `admin_year` | the period. `201800`–`202400` are calendar years; from `202501` the last two digits are the quarter, so `202602` is 2026 Q2 |
| `intersected_area_ha` | hectares of mining **first detected** in this period |
| `intersected_area_ha_cumulative` | hectares detected from 2018 **through** this period |
| `status` | designation status for Indigenous territories and protected areas, in the source's own wording and language. Blank for admin areas and wherever the source gives none |
| `bbox_minx` ... `bbox_maxy` | jurisdiction bounding box, EPSG:4326 |


Areas are derived from segmentation, which overshoots hand-annotated extent — read the caveat in the
[repo README](../../README.md) before quoting a total.

## `illegality_analysis_by_jurisdiction.csv`

The same jurisdictions, with their mined area split by how likely that mining is to be illegal. **One row per jurisdiction, not per period.** The split is published for the latest period only, so there is no timeseries to give.

Four bands:

| band | `admin_illegality_max` | columns |
| --- | --- | --- |
| Very high | 4 | `illegality_very_high_affected_area_ha`, `..._pct` |
| High | 3 | `illegality_high_affected_area_ha`, `..._pct` |
| Medium | 2 | `illegality_medium_affected_area_ha`, `..._pct` |
| Low | 1 | `illegality_low_affected_area_ha`, `..._pct` |

The `_ha` columns hold hectares of detected mining falling in that band; the `_pct` columns hold that band's share of the jurisdiction's mined area, as a fraction between 0 and 1.

| column | meaning |
| --- | --- |
| `date_published`, `id`, `name`, `country`, `country_code`, `type`, `status`, `bbox_minx` ... `bbox_maxy` | the jurisdiction, exactly as in `mined_areas_by_jurisdiction.csv` above |
| `admin_year` | which period the split describes, in the same encoding as the other file. Almost always the latest published period; a jurisdiction whose timeseries stops earlier carries its own last period instead, so check this column before treating the file as one snapshot |
| `illegality_*_affected_area_ha` | hectares of detected mining in that band |
| `illegality_*_affected_area_pct` | that band's share of the jurisdiction's mined area, 0–1 |

`0` and a blank mean different things: `0` is a measured absence (no mined area
fell in that band) while a blank row means no illegality data was available for that jurisdiction.
