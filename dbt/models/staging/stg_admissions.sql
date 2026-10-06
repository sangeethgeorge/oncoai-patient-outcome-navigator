select
    hadm_id,
    admittime,
    dischtime,
    deathtime,
    cast(hospital_expire_flag as integer) as hospital_expire_flag
from {{ source('mimic', 'admissions') }}
