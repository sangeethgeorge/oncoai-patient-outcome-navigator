-- Readmission is undefined for patients who died in the ICU; survivors always get true/false.
select icustay_id
from {{ ref('int_icu_utilization') }}
where (icu_death and (icu_readmit_48h is not null or icu_readmit_same_adm is not null))
   or (not icu_death and (icu_readmit_48h is null or icu_readmit_same_adm is null))
   or (icu_readmit_48h and not icu_readmit_same_adm)
