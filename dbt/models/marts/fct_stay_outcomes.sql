-- One row per cohort ICU stay: who, what cancer, where, how long, what happened, and the
-- cross-fitted predicted risk (for risk tiers and subgroup calibration checks).
select
    c.icustay_id,
    c.subject_id,
    c.hadm_id,
    c.age,
    c.gender,
    c.admission_type,
    c.insurance,
    dx.cancer_group,
    dx.primary_site,
    dx.heme_subtype,
    dx.metastatic_flag,
    dx.hematologic_flag,
    dx.hematologic_and_solid_flag,
    dx.n_cancer_codes,
    u.first_careunit,
    u.dbsource,
    u.icu_los_days,
    u.hosp_los_days,
    u.icu_death,
    u.icu_readmit_48h,
    u.icu_readmit_same_adm,
    u.hospital_expire_flag,
    c.mortality_30d,
    p.pred_prob,
    p.cv_fold
from {{ ref('stg_cohort') }} c
left join {{ ref('int_stay_cancer_dx') }} dx using (icustay_id)
left join {{ ref('int_icu_utilization') }} u using (icustay_id)
left join {{ ref('stg_oof_predictions') }} p using (icustay_id)
