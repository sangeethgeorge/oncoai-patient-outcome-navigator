DROP MATERIALIZED VIEW IF EXISTS all_vitals_labs_48h;

CREATE MATERIALIZED VIEW all_vitals_labs_48h AS
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
-- Keep only vitals that appear in ≥80% of ICU stays
vital_valid_items AS (
    SELECT vc.itemid
    FROM vital_counts vc, total_cohort tc
    WHERE vc.icu_count >= 0.80 * tc.total_icu
),
-- Keep only labs that appear in ≥80% of ICU stays
lab_valid_items AS (
    SELECT lc.itemid
    FROM lab_counts lc, total_cohort tc
    WHERE lc.icu_count >= 0.80 * tc.total_icu
)

-- Final view: all qualifying vitals and labs in first 48h
SELECT
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
	c.icd9_code,
	c.short_title,
    'vital' AS source,
    ce.itemid,
    di.label,
    ce.charttime,
    ce.valuenum
FROM chartevents ce
JOIN cohort c ON ce.icustay_id = c.icustay_id
JOIN d_items di ON ce.itemid = di.itemid
JOIN vital_valid_items vi ON ce.itemid = vi.itemid
WHERE ce.valuenum IS NOT NULL
  AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '48 hours'

UNION ALL

SELECT
    c.subject_id,
    c.hadm_id,
    c.icustay_id,
	c.icd9_code,
	c.short_title,
    'lab' AS source,
    le.itemid,
    dl.label,
    le.charttime,
    le.valuenum
FROM labevents le
JOIN cohort c ON le.subject_id = c.subject_id AND le.hadm_id = c.hadm_id
JOIN d_labitems dl ON le.itemid = dl.itemid
JOIN lab_valid_items li ON le.itemid = li.itemid
WHERE le.valuenum IS NOT NULL
  AND le.charttime BETWEEN c.intime AND c.intime + INTERVAL '48 hours'
  
WITH DATA;
