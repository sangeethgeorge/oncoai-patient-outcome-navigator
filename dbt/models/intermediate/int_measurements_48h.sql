-- Labs and chart vitals in one long table with LOINC codes.
-- Labs: LOINC from MIMIC's d_labitems. Chart items: MIMIC-III has no LOINC, so a small
-- hand-verified crosswalk (seed chart_item_loinc) covers the vitals the model uses.
select
    'lab' as source,
    l.icustay_id,
    l.itemid,
    l.label,
    l.charttime,
    l.value,
    l.value_uom,
    li.loinc_code,
    case when li.loinc_code is not null then 'mimic_d_labitems' end as loinc_source
from {{ ref('stg_labs_48h') }} l
left join {{ ref('stg_lab_items') }} li using (itemid)

union all

select
    'chart' as source,
    v.icustay_id,
    v.itemid,
    v.label,
    v.charttime,
    v.value,
    v.value_uom,
    x.loinc_code,
    case when x.loinc_code is not null then 'crosswalk_seed' end as loinc_source
from {{ ref('stg_vitals_48h') }} v
left join {{ ref('chart_item_loinc') }} x using (itemid)
