-- Data dictionary for every lab and chart item seen in the 48 h window: coding, units, volume.
with cohort_n as (select count(*) as n from {{ ref('stg_cohort') }})

select
    m.source,
    m.itemid,
    m.label,
    any_value(m.loinc_code) as loinc_code,
    any_value(m.loinc_source) as loinc_source,
    count(distinct m.value_uom) as n_units,
    string_agg(distinct m.value_uom, ', ') as units,
    count(*) as n_rows,
    count(distinct m.icustay_id) as n_stays,
    count(distinct m.icustay_id) / any_value(cohort_n.n) as pct_stays
from {{ ref('int_measurements_48h') }} m
cross join cohort_n
group by m.source, m.itemid, m.label
