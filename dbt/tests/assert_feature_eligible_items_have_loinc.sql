-- Every measurement eligible as a model feature must be LOINC-coded (lab dictionary or verified
-- crosswalk). Eligibility mirrors feature_engineering.py, which pools items by label: labs in >= 70%
-- of stays, chart items in >= 95%.
with by_label as (
    select
        source,
        label,
        count(distinct icustay_id) / (select count(*) from {{ ref('stg_cohort') }}) as pct_stays,
        count(*) filter (where loinc_code is null) as n_uncoded_rows
    from {{ ref('int_measurements_48h') }}
    group by source, label
)

select *
from by_label
where n_uncoded_rows > 0
  and ((source = 'lab' and pct_stays >= 0.70) or (source = 'chart' and pct_stays >= 0.95))
