{{ config(severity='warn') }}
-- Items recorded in more than one unit need a conversion rule before their values are pooled.
-- Case-only differences ('BPM' vs 'bpm') are ignored.
select source, itemid, label, units
from {{ ref('dim_measurement_item') }}
where (select count(distinct lower(u)) from unnest(string_split(units, ', ')) t(u)) > 1
