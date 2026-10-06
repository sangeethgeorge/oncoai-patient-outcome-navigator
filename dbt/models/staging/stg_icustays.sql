select
    icustay_id,
    subject_id,
    hadm_id,
    dbsource,
    first_careunit,
    intime,
    outtime
from {{ source('mimic', 'icustays') }}
