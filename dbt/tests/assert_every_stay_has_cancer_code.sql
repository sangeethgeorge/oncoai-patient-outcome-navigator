-- The cohort rule and the category seed must agree: every stay maps to at least one category.
select c.icustay_id
from {{ ref('stg_cohort') }} c
left join {{ ref('int_stay_cancer_dx') }} dx using (icustay_id)
where dx.icustay_id is null
