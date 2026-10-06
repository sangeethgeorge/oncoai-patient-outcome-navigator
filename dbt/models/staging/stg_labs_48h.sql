select
    icustay_id,
    itemid,
    labs_label as label,
    fluid,
    charttime,
    labs_valuenum as value,
    labs_valueom as value_uom  -- typo in the extraction view's column name
from {{ source('mimic', 'all_labs_48h') }}
