-- Length of stay, ICU death and ICU readmission for each cohort stay.
-- ICU death: in-hospital death no later than 6 h after ICU discharge (MIMIC closes some stays
-- shortly after death is charted). Readmission is defined only for ICU survivors, following the
-- SCCM indicator: another ICU stay in the same hospital admission within 48 h of ICU discharge.
with stay as (
    select
        c.icustay_id,
        c.hadm_id,
        c.intime,
        c.outtime,
        i.first_careunit,
        i.dbsource,
        a.admittime,
        a.dischtime,
        a.hospital_expire_flag,
        a.deathtime is not null and a.deathtime <= c.outtime + interval 6 hour as icu_death
    from {{ ref('stg_cohort') }} c
    join {{ ref('stg_icustays') }} i using (icustay_id)
    join {{ ref('stg_admissions') }} a on a.hadm_id = c.hadm_id
),

next_stay as (
    select s.icustay_id, min(i.intime) as next_icu_intime
    from stay s
    join {{ ref('stg_icustays') }} i
        on i.hadm_id = s.hadm_id and i.icustay_id <> s.icustay_id and i.intime >= s.outtime
    group by s.icustay_id
)

select
    s.icustay_id,
    s.first_careunit,
    s.dbsource,
    epoch(s.outtime - s.intime) / 86400.0 as icu_los_days,
    epoch(s.dischtime - s.admittime) / 86400.0 as hosp_los_days,
    s.hospital_expire_flag,
    s.icu_death,
    case when not s.icu_death then n.next_icu_intime is not null end as icu_readmit_same_adm,
    case when not s.icu_death
         then coalesce(n.next_icu_intime <= s.outtime + interval 48 hour, false) end as icu_readmit_48h
from stay s
left join next_stay n using (icustay_id)
