-- One row per cohort stay with its cancer group and primary site, from the admission's ICD-9 codes
-- (seed icd9_cancer_category). Grouping follows the critical-care oncology literature:
--   hematologic          any lymphoma/leukemia/myeloma code (200-208), whatever else is coded
--   solid_metastatic     otherwise, any secondary/disseminated code (196-199.1, 209.7)
--   solid_nonmetastatic  everything else
-- primary_site: category of the lowest seq_num primary-site code; 'unknown_primary' when the only
-- cancer codes are secondary.
-- heme_subtype (hematologic stays only), first match wins: acute leukemia (204.0, 205.0, 206.0, 207.0,
-- 208.0), lymphoma (200-202), myeloma (203), chronic leukemia (204.1-208.1), other.
with coded as (
    select
        c.icustay_id,
        d.seq_num,
        d.icd9_code,
        cat.site_category,
        cat.is_hematologic,
        cat.is_secondary
    from {{ ref('stg_cohort') }} c
    join {{ ref('stg_diagnoses') }} d on d.hadm_id = c.hadm_id
    join {{ ref('icd9_cancer_category') }} cat on d.icd4 between cat.icd4_low and cat.icd4_high
),

flags as (
    select
        icustay_id,
        bool_or(is_hematologic) as hematologic_flag,
        bool_or(is_secondary) as metastatic_flag,
        bool_or(not is_hematologic and not is_secondary) as solid_primary_flag,
        coalesce(arg_min(site_category, seq_num) filter (where not is_secondary), 'unknown_primary') as primary_site,
        count(distinct icd9_code) as n_cancer_codes,
        bool_or(substr(icd9_code, 1, 4) in ('2040', '2050', '2060', '2070', '2080')) as acute_leukemia,
        bool_or(substr(icd9_code, 1, 3) in ('200', '201', '202')) as lymphoma,
        bool_or(substr(icd9_code, 1, 3) = '203') as myeloma,
        bool_or(substr(icd9_code, 1, 4) in ('2041', '2051', '2061', '2071', '2081')) as chronic_leukemia
    from coded
    group by icustay_id
)

select
    icustay_id,
    case
        when hematologic_flag then 'hematologic'
        when metastatic_flag then 'solid_metastatic'
        else 'solid_nonmetastatic'
    end as cancer_group,
    primary_site,
    case
        when not hematologic_flag then null
        when acute_leukemia then 'acute_leukemia'
        when lymphoma then 'lymphoma'
        when myeloma then 'myeloma'
        when chronic_leukemia then 'chronic_leukemia'
        else 'other_hematologic'
    end as heme_subtype,
    hematologic_flag,
    metastatic_flag,
    hematologic_flag and solid_primary_flag as hematologic_and_solid_flag,
    n_cancer_codes
from flags
