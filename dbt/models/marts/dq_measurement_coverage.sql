-- Data-health monitor. For each measurement (grouped by LOINC, so CareVue and MetaVision items
-- line up) and each slice (care unit, documentation system), the share of stays with any value and
-- the share of values outside wide plausibility limits. MIMIC dates are shifted per patient, so
-- calendar trends are meaningless; drift is checked across units and systems instead.
{% set coverage_gap = 0.15 %}
{% set max_implausible = 0.01 %}

with stays as (
    select icustay_id, 'first_careunit' as slice_type, first_careunit as slice from {{ ref('int_icu_utilization') }}
    union all
    select icustay_id, 'dbsource', dbsource from {{ ref('int_icu_utilization') }}
),

measured as (
    select m.icustay_id, m.loinc_code, m.label, m.value, r.low, r.high
    from {{ ref('int_measurements_48h') }} m
    join {{ ref('measurement_plausible_range') }} r using (loinc_code)
),

slice_sizes as (
    select slice_type, slice, count(*) as n_stays from stays group by all
),

by_slice as (
    select
        s.slice_type,
        s.slice,
        m.loinc_code,
        any_value(m.label) as measurement,
        count(distinct m.icustay_id) as n_stays_measured,
        count(*) as n_values,
        avg(case when m.value < m.low or m.value > m.high then 1.0 else 0.0 end) as pct_implausible
    from stays s
    join measured m using (icustay_id)
    group by all
),

overall as (
    select loinc_code,
           count(distinct icustay_id) / (select count(*) from {{ ref('stg_cohort') }}) as overall_coverage
    from measured
    group by loinc_code
)

select
    b.slice_type,
    b.slice,
    b.loinc_code,
    b.measurement,
    z.n_stays,
    b.n_stays_measured,
    b.n_stays_measured / z.n_stays as coverage,
    o.overall_coverage,
    b.n_values,
    b.pct_implausible,
    abs(b.n_stays_measured / z.n_stays - o.overall_coverage) >= {{ coverage_gap }} as flag_coverage_gap,
    b.pct_implausible > {{ max_implausible }} as flag_implausible_values
from by_slice b
join slice_sizes z using (slice_type, slice)
join overall o using (loinc_code)
