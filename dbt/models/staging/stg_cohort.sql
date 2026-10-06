select
    subject_id,
    hadm_id,
    icustay_id,
    intime,
    outtime,
    admittime,
    dod,
    gender,
    cast(age as integer) as age,
    admission_type,
    insurance,
    cast(mortality_30d as integer) as mortality_30d
from {{ source('mimic', 'oncology_icu_base') }}
