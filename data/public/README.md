# Publicly served files

These paths are public URLs, linked straight from the Amazon Mining Watch website.

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
| `bbox_minx` … `bbox_maxy` | jurisdiction bounding box, EPSG:4326 |


Areas are derived from segmentation, which overshoots hand-annotated extent — read the caveat in the
[repo README](../../README.md) before quoting a total.
