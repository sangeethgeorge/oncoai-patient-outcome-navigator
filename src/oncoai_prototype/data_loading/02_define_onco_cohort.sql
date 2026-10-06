-- Oncology ICU cohort: one row per ICU stay.
--   * first ICU stay per patient, in an admission with a malignant/neoplasm ICD-9 code (140-239)
--   * ICU stay >= 48h, so the 48h feature window ends before the outcome can occur in the ICU
--   * adults 18-89 (MIMIC shifts ages > 89)
--   * outcome: death within 30 days of ICU admission (intime), the prediction anchor
DROP MATERIALIZED VIEW IF EXISTS oncology_icu_base CASCADE;

CREATE MATERIALIZED VIEW oncology_icu_base AS
WITH oncology_diagnoses AS (
    -- Aggregate to one row per admission: patients often carry several cancer codes,
    -- and joining them unaggregated duplicated ICU stays.
    SELECT di.subject_id, di.hadm_id,
           ARRAY_AGG(DISTINCT di.icd9_code ORDER BY di.icd9_code) AS icd9_codes,
           COUNT(DISTINCT di.icd9_code) AS n_cancer_codes
    FROM diagnoses_icd di
    JOIN d_icd_diagnoses dd ON di.icd9_code = dd.icd9_code
    WHERE
        dd.icd9_code ~ '^[0-9]{3}' -- exclude E/V codes
        AND CAST(SUBSTRING(dd.icd9_code FROM 1 FOR 3) AS INTEGER) BETWEEN 140 AND 239
        AND dd.long_title ILIKE ANY (ARRAY[
            '%malignant%',
            '%neoplasm%'
        ])
    GROUP BY di.subject_id, di.hadm_id
),
first_icu_stays AS (
    SELECT icu.subject_id, icu.hadm_id, icu.icustay_id, icu.intime, icu.outtime,
           ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS rn
    FROM icustays icu
),
oncology_icu_cohort AS (
    SELECT f.subject_id, f.hadm_id, f.icustay_id, f.intime, f.outtime,
           o.icd9_codes, o.n_cancer_codes
    FROM first_icu_stays f
    JOIN oncology_diagnoses o ON f.subject_id = o.subject_id AND f.hadm_id = o.hadm_id
    WHERE f.rn = 1
      AND f.outtime - f.intime >= INTERVAL '48 hours'
)
SELECT
    o.subject_id,
    o.hadm_id,
    o.icustay_id,
    o.icd9_codes,
    o.n_cancer_codes,
    o.intime,
    o.outtime,
    p.gender,
    p.dob,
    p.dod,
    a.admittime,
    a.ethnicity,
    a.marital_status,
    a.insurance,
    a.admission_type,
    FLOOR(EXTRACT(EPOCH FROM (a.admittime - p.dob))/31557600) AS age,
    CASE
        WHEN p.dod IS NOT NULL AND p.dod <= o.intime + INTERVAL '30 days' THEN 1
        ELSE 0
    END AS mortality_30d
FROM oncology_icu_cohort o
JOIN admissions a ON o.hadm_id = a.hadm_id
JOIN patients p ON o.subject_id = p.subject_id
WHERE FLOOR(EXTRACT(EPOCH FROM (a.admittime - p.dob))/31557600) BETWEEN 18 AND 89
WITH DATA;

CREATE UNIQUE INDEX oncology_icu_base_icustay_idx ON oncology_icu_base (icustay_id);
