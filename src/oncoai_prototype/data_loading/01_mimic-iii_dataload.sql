-- Loads MIMIC-III v1.4 CSVs into Postgres.
-- Server-side COPY needs an absolute path, passed as a psql variable:
--   psql "$ONCOAI_POSTGRES_CONN_STR" -v mimic_dir="$PWD/data/raw/mimic-iii-full" \
--        -f src/oncoai_prototype/data_loading/01_mimic-iii_dataload.sql
-- --- PATIENTS Table ---
DROP TABLE IF EXISTS PATIENTS; -- Safely drops the table if it exists

CREATE TABLE PATIENTS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
    ROW_ID INT PRIMARY KEY,
	SUBJECT_ID INT,
	GENDER VARCHAR(5),
	DOB	TIMESTAMP(0),
	DOD	TIMESTAMP(0),
	DOD_HOSP TIMESTAMP(0),
	DOD_SSN	TIMESTAMP(0),
	EXPIRE_FLAG	VARCHAR(5)
);

\set csv_file :mimic_dir '/PATIENTS.csv/PATIENTS.csv'
COPY PATIENTS FROM :'csv_file' WITH (FORMAT csv, HEADER true);

DROP TABLE IF EXISTS ADMISSIONS; -- Safely drops the table if it exists

-- --- ADMISSIONS Table ---
CREATE TABLE ADMISSIONS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
    ROW_ID	INT PRIMARY KEY,
	SUBJECT_ID	INT,
	HADM_ID	INT,
	ADMITTIME	TIMESTAMP(0),
	DISCHTIME	TIMESTAMP(0),
	DEATHTIME	TIMESTAMP(0),
	ADMISSION_TYPE	VARCHAR(50),
	ADMISSION_LOCATION	VARCHAR(50),
	DISCHARGE_LOCATION	VARCHAR(50),
	INSURANCE	VARCHAR(255),
	LANGUAGE	VARCHAR(10),
	RELIGION	VARCHAR(50),
	MARITAL_STATUS	VARCHAR(50),
	ETHNICITY	VARCHAR(200),
	EDREGTIME	TIMESTAMP(0),
	EDOUTTIME	TIMESTAMP(0),
	DIAGNOSIS	VARCHAR(300),
	HOSPITAL_EXPIRE_FLAG	SMALLINT,	--Modified TINYINT to SMALLINT
	HAS_CHARTEVENTS_DATA	SMALLINT		--Modified TINYINT to SMALLINT
);

\set csv_file :mimic_dir '/ADMISSIONS.csv/ADMISSIONS.csv'
COPY ADMISSIONS FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- ICUSTAYS Table ---
DROP TABLE IF EXISTS ICUSTAYS; -- Safely drops the table if it exists

CREATE TABLE ICUSTAYS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,
	SUBJECT_ID	INT,
	HADM_ID	INT,
	ICUSTAY_ID	INT,
	DBSOURCE	VARCHAR(20),
	FIRST_CAREUNIT	VARCHAR(20),
	LAST_CAREUNIT	VARCHAR(20),
	FIRST_WARDID	SMALLINT,
	LAST_WARDID	SMALLINT,
	INTIME	TIMESTAMP(0),
	OUTTIME	TIMESTAMP(0),
	LOS	DOUBLE PRECISION
);

\set csv_file :mimic_dir '/ICUSTAYS.csv/ICUSTAYS.csv'
COPY ICUSTAYS FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- DIAGNOSES_ICD Table ---
DROP TABLE IF EXISTS DIAGNOSES_ICD; -- Safely drops the table if it exists

CREATE TABLE DIAGNOSES_ICD (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT	not null PRIMARY KEY,
	SUBJECT_ID	INT	not null,
	HADM_ID	INT	not null,
	SEQ_NUM	INT,	
	ICD9_CODE	VARCHAR(10)
);

\set csv_file :mimic_dir '/DIAGNOSES_ICD.csv/DIAGNOSES_ICD.csv'
COPY DIAGNOSES_ICD FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- D_ICD_DIAGNOSES Table ---
DROP TABLE IF EXISTS D_ICD_DIAGNOSES; -- Safely drops the table if it exists

CREATE TABLE D_ICD_DIAGNOSES (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,
	ICD9_CODE	VARCHAR(10),
	SHORT_TITLE	VARCHAR(50),
	LONG_TITLE	VARCHAR(300)
);

\set csv_file :mimic_dir '/D_ICD_DIAGNOSES.csv/D_ICD_DIAGNOSES.csv'
COPY D_ICD_DIAGNOSES FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- CHARTEVEVENTS Table ---
DROP TABLE IF EXISTS CHARTEVENTS; -- Safely drops the table if it exists

CREATE TABLE CHARTEVENTS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,		-- CHARTEVENTS.csv is a huge file, will take long to readd (~35Gb)
	SUBJECT_ID	NUMERIC(7,0),
	HADM_ID	NUMERIC(7,0),
	ICUSTAY_ID	NUMERIC(7,0),
	ITEMID	NUMERIC(7,0),
	CHARTTIME	TIMESTAMP(0),	-- was DATE, which dropped the time of day
	STORETIME	TIMESTAMP(0),
	CGID	NUMERIC(7,0),
	VALUE	VARCHAR(200),
	VALUENUM	NUMERIC,
	VALUEUOM	VARCHAR(200),
	WARNING	NUMERIC(1,0),
	ERROR	NUMERIC(1,0),
	RESULTSTATUS	VARCHAR(200),
	STOPPED	VARCHAR(200)
);

\set csv_file :mimic_dir '/CHARTEVENTS.csv/CHARTEVENTS.csv'
COPY CHARTEVENTS FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- D_ITEMS Table ---
DROP TABLE IF EXISTS D_ITEMS; -- Safely drops the table if it exists

CREATE TABLE D_ITEMS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,
	ITEMID	INT,
	LABEL	VARCHAR(200),
	ABBREVIATION	VARCHAR(100),
	DBSOURCE	VARCHAR(20),
	LINKSTO	VARCHAR(50),
	CATEGORY	VARCHAR(100),
	UNITNAME	VARCHAR(100),
	PARAM_TYPE	VARCHAR(30),
	CONCEPTID	INT
);

\set csv_file :mimic_dir '/D_ITEMS.csv/D_ITEMS.csv'
COPY D_ITEMS FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- LABEVENTS Table ---
DROP TABLE IF EXISTS LABEVENTS; -- Safely drops the table if it exists

CREATE TABLE LABEVENTS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,
	SUBJECT_ID	INT,
	HADM_ID	INT,
	ITEMID	INT,
	CHARTTIME	TIMESTAMP(0),
	VALUE	VARCHAR(200),
	VALUENUM	DOUBLE PRECISION,
	VALUEUOM	VARCHAR(20),
	FLAG	VARCHAR(20)
);

\set csv_file :mimic_dir '/LABEVENTS.csv/LABEVENTS.csv'
COPY LABEVENTS FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- NOTEEVENTS Table ---
DROP TABLE IF EXISTS NOTEEVENTS; -- Safely drops the table if it exists


CREATE TABLE NOTEEVENTS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,
	SUBJECT_ID	INT,
	HADM_ID	INT,
	CHARTDATE	TIMESTAMP(0),
	CHARTTIME	TIMESTAMP(0),
	STORETIME	TIMESTAMP(0),
	CATEGORY	VARCHAR(50),
	DESCRIPTION	VARCHAR(300),
	CGID	INT,
	ISERROR	CHAR(1),
	TEXT	TEXT
);

\set csv_file :mimic_dir '/NOTEEVENTS.csv/NOTEEVENTS.csv'
COPY NOTEEVENTS FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- D_LABITEMS Table ---
DROP TABLE IF EXISTS D_LABITEMS; -- Safely drops the table if it exists


CREATE TABLE D_LABITEMS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,
	ITEMID	INT,
	LABEL	VARCHAR(100),
	FLUID	VARCHAR(100),
	CATEGORY	VARCHAR(100),
	LOINC_CODE	VARCHAR(100)
	);

\set csv_file :mimic_dir '/D_LABITEMS.csv/D_LABITEMS.csv'
COPY D_LABITEMS FROM :'csv_file' WITH (FORMAT csv, HEADER true);

-- --- Indexes for the cohort and 48h extraction views ---
CREATE INDEX IF NOT EXISTS chartevents_icustay_idx ON CHARTEVENTS (ICUSTAY_ID);
CREATE INDEX IF NOT EXISTS labevents_subject_hadm_idx ON LABEVENTS (SUBJECT_ID, HADM_ID);
