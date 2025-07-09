DROP MATERIALIZED VIEW IF EXISTS all_vitals_48h;

CREATE MATERIALIZED VIEW all_vitals_48h AS
WITH cohort AS (
    SELECT * FROM oncology_icu_base
)
SELECT
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
    c.icd9_code,
    c.short_title,
    ce.itemid,
    di.label AS vitals_label,
    di.category AS vitals_category,
    di.dbsource,
    ce.charttime,
    ce.valuenum AS vitals_valuenum,
    ce.valueuom AS vitals_valueuom
FROM chartevents ce
JOIN cohort c ON ce.icustay_id = c.icustay_id
JOIN d_items di ON ce.itemid = di.itemid
WHERE ce.valuenum IS NOT NULL
  AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '48 hours';
