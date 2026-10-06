select
    cast(icustay_id as integer) as icustay_id,
    cast(fold as integer) as cv_fold,
    pred_prob
from {{ source('modeling', 'oof_predictions') }}
