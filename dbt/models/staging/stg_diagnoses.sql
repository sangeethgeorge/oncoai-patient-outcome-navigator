-- ICD-9-CM diagnosis codes are stored without the decimal point ('1749' = 174.9, '185' = 185).
-- icd4 puts every numeric code on one integer scale (first three digits * 10 + fourth digit)
-- so category ranges can be joined with BETWEEN. E and V codes get a null icd4.
select
    subject_id,
    hadm_id,
    seq_num,
    icd9_code,
    case
        when regexp_matches(icd9_code, '^[0-9]{3}')
            then cast(substr(icd9_code, 1, 3) as integer) * 10
                 + coalesce(try_cast(substr(icd9_code, 4, 1) as integer), 0)
    end as icd4
from {{ source('mimic', 'diagnoses_icd') }}
