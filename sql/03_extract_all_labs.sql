DROP MATERIALIZED VIEW IF EXISTS all_labs_48h;

CREATE MATERIALIZED VIEW all_labs_48h AS
WITH cohort AS (
    SELECT * FROM oncology_icu_base
),
-- Count how many ICU stays each lab item appears in (via subject_id + hadm_id)
lab_counts AS (
    SELECT le.itemid, COUNT(DISTINCT c.icustay_id) AS icu_count
    FROM labevents le
    JOIN cohort c ON le.subject_id = c.subject_id AND le.hadm_id = c.hadm_id
    WHERE le.valuenum IS NOT NULL
      AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '48 hours'
    GROUP BY le.itemid
),
-- Get total number of ICU stays in the cohort
total_cohort AS (
    SELECT COUNT(DISTINCT icustay_id) AS total_icu FROM cohort
),
-- Keep only labs that appear in ≥80% of ICU stays
lab_valid_items AS (
    SELECT lc.itemid
    FROM lab_counts lc, total_cohort tc
)

-- Final view: all qualifying labs in first 48h
SELECT
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
    c.icd9_code,
    c.short_title,
    le.itemid,
    dl.label,
    le.charttime,
    le.valuenum,
    le.valueuom 
FROM labevents le
JOIN cohort c ON le.subject_id = c.subject_id AND le.hadm_id = c.hadm_id
JOIN d_labitems dl ON le.itemid = dl.itemid
JOIN lab_valid_items li ON le.itemid = li.itemid
WHERE le.valuenum IS NOT NULL
  AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '48 hours'
WITH DATA;