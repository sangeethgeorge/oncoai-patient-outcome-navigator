DROP MATERIALIZED VIEW IF EXISTS all_vitals_48h;

CREATE MATERIALIZED VIEW all_vitals_48h AS
WITH cohort AS (
    SELECT * FROM oncology_icu_base
),
-- Count how many ICU stays each vital item appears in
vital_counts AS (
    SELECT ce.itemid, COUNT(DISTINCT ce.icustay_id) AS icu_count
    FROM chartevents ce
    JOIN cohort c ON ce.icustay_id = c.icustay_id
    WHERE ce.valuenum IS NOT NULL
      AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '48 hours'
    GROUP BY ce.itemid
),
-- Get total number of ICU stays in the cohort
total_cohort AS (
    SELECT COUNT(DISTINCT icustay_id) AS total_icu FROM cohort
),
-- Keep only vitals that appear in ≥80% of ICU stays
vital_valid_items AS (
    SELECT vc.itemid
    FROM vital_counts vc, total_cohort tc
)

-- Final view: all qualifying vitals in first 48h
SELECT
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
    c.icd9_code,
    c.short_title,
    ce.itemid,
    di.label,
    ce.charttime,
    ce.valuenum,
    ce.valueuom -- Added valueuom for vitals
FROM chartevents ce
JOIN cohort c ON ce.icustay_id = c.icustay_id
JOIN d_items di ON ce.itemid = di.itemid
JOIN vital_valid_items vi ON ce.itemid = vi.itemid
WHERE ce.valuenum IS NOT NULL
  AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '48 hours'
WITH DATA;