DROP MATERIALIZED VIEW IF EXISTS all_labs_48h;

CREATE MATERIALIZED VIEW all_labs_48h AS
WITH cohort AS (
    SELECT * FROM oncology_icu_base
)
SELECT
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
    c.icd9_code,
    c.short_title,
    le.itemid,
    dl.label AS labs_label,
    dl.fluid,
    dl.category AS labs_category,
    le.charttime,
    le.valuenum AS labs_valuenum,
    le.valueuom AS labs_valueom,
    le.flag
FROM labevents le
JOIN cohort c ON le.subject_id = c.subject_id AND le.hadm_id = c.hadm_id
JOIN d_labitems dl ON le.itemid = dl.itemid
WHERE le.valuenum IS NOT NULL
  AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '48 hours';
