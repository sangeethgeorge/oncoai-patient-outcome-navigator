DROP MATERIALIZED VIEW IF EXISTS oncology_icu_base;

CREATE MATERIALIZED VIEW oncology_icu_base AS
WITH oncology_diagnoses AS (
    SELECT DISTINCT di.subject_id, di.hadm_id, di.icd9_code, dd.short_title
    FROM diagnoses_icd di
    JOIN d_icd_diagnoses dd ON di.icd9_code = dd.icd9_code
    WHERE
        dd.icd9_code ~ '^[0-9]{3}' -- exclude E/V codes
        AND CAST(SUBSTRING(dd.icd9_code FROM 1 FOR 3) AS INTEGER) BETWEEN 140 AND 239
        AND dd.long_title ILIKE ANY (ARRAY[
            '%malignant%',
            '%neoplasm%'
        ])
),
first_icu_stays AS (
    SELECT icu.subject_id, icu.hadm_id, icu.icustay_id, icu.intime, icu.outtime,
           ROW_NUMBER() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) AS rn
    FROM icustays icu
),
oncology_icu_cohort AS (
    SELECT f.subject_id, f.hadm_id, f.icustay_id, f.intime, f.outtime,
			o.icd9_code, o.short_title
    FROM first_icu_stays f
    JOIN oncology_diagnoses o ON f.subject_id = o.subject_id AND f.hadm_id = o.hadm_id
    WHERE f.rn = 1
      AND EXTRACT(EPOCH FROM (f.outtime - f.intime))/3600 <= 48 -- ICU stay ≤ 48h
),
patient_info AS (
    SELECT p.subject_id, p.dob, p.gender, p.dod,
           a.hadm_id, a.admittime, a.ethnicity, a.marital_status,
           a.insurance, a.admission_type
    FROM patients p
    JOIN admissions a ON p.subject_id = a.subject_id
)
SELECT
    o.subject_id,
    o.hadm_id,
    o.icustay_id,
	o.icd9_code, 
	o.short_title,
    o.intime,
    o.outtime,
    pi.gender,
    pi.dob,
    pi.dod,
    pi.admittime,
    pi.ethnicity,
    pi.marital_status,
    pi.insurance,
    pi.admission_type,
    FLOOR(EXTRACT(EPOCH FROM (pi.admittime - pi.dob))/31557600) AS age,
    CASE 
        WHEN pi.dod IS NOT NULL AND DATE_PART('day', pi.dod - pi.admittime) <= 30 THEN 1
        ELSE 0
    END AS mortality_30d
FROM oncology_icu_cohort o
JOIN patient_info pi ON o.subject_id = pi.subject_id AND o.hadm_id = pi.hadm_id
WHERE FLOOR(EXTRACT(EPOCH FROM (pi.admittime - pi.dob))/31557600) BETWEEN 18 AND 89
WITH DATA;
