select
    itemid,
    label,
    fluid,
    category,
    nullif(trim(loinc_code), '') as loinc_code
from {{ source('mimic', 'd_labitems') }}
