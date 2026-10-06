-- Internal consistency: a death in hospital must carry a date of death.
select o.icustay_id
from {{ ref('fct_stay_outcomes') }} o
join {{ ref('stg_cohort') }} c using (icustay_id)
where o.hospital_expire_flag = 1 and c.dod is null
