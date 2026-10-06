-- O/E is only valid if every cohort stay has exactly one cross-fitted prediction and no extras.
select coalesce(c.icustay_id, p.icustay_id) as icustay_id
from {{ ref('stg_cohort') }} c
full outer join {{ ref('stg_oof_predictions') }} p using (icustay_id)
where c.icustay_id is null or p.icustay_id is null
