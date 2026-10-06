-- Every lab/chart value must fall in [intime, intime + 48h) of its stay.
select m.source, m.icustay_id, m.charttime
from {{ ref('int_measurements_48h') }} m
join {{ ref('stg_cohort') }} c using (icustay_id)
where m.charttime < c.intime or m.charttime >= c.intime + interval 48 hour
