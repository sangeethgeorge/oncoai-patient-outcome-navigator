select
    icustay_id,
    itemid,
    vitals_label as label,
    dbsource,
    charttime,
    vitals_valuenum as value,
    vitals_valueuom as value_uom
from {{ source('mimic', 'all_vitals_48h') }}
