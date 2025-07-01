-- This SQL script loads the .csv files into tables from mimic-iii 
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

COPY PATIENTS FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/PATIENTS.csv/PATIENTS.csv' WITH (FORMAT csv, HEADER true);

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

COPY ADMISSIONS FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/ADMISSIONS.csv/ADMISSIONS.csv' WITH (FORMAT csv, HEADER true);

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

COPY ICUSTAYS FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/ICUSTAYS.csv/ICUSTAYS.csv' WITH (FORMAT csv, HEADER true);

-- --- DIAGNOSES_ICD Table ---
DROP TABLE IF EXISTS DIAGNOSES_ICD; -- Safely drops the table if it exists

CREATE TABLE DIAGNOSES_ICD (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT	not null PRIMARY KEY,
	SUBJECT_ID	INT	not null,
	HADM_ID	INT	not null,
	SEQ_NUM	INT,	
	ICD9_CODE	VARCHAR(10)
);

COPY DIAGNOSES_ICD FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/DIAGNOSES_ICD.csv/DIAGNOSES_ICD.csv' WITH (FORMAT csv, HEADER true);

-- --- D_ICD_DIAGNOSES Table ---
DROP TABLE IF EXISTS D_ICD_DIAGNOSES; -- Safely drops the table if it exists

CREATE TABLE D_ICD_DIAGNOSES (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,
	ICD9_CODE	VARCHAR(10),
	SHORT_TITLE	VARCHAR(50),
	LONG_TITLE	VARCHAR(300)
);

COPY D_ICD_DIAGNOSES FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/D_ICD_DIAGNOSES.csv/D_ICD_DIAGNOSES.csv' WITH (FORMAT csv, HEADER true);

-- --- CHARTEVEVENTS Table ---
DROP TABLE IF EXISTS CHARTEVENTS; -- Safely drops the table if it exists

CREATE TABLE CHARTEVENTS (			-- Table Schema from "https://mimic.mit.edu/docs/iii/tables/"
	ROW_ID	INT PRIMARY KEY,		-- CHARTEVENTS.csv is a huge file, will take long to readd (~35Gb)
	SUBJECT_ID	NUMERIC(7,0),
	HADM_ID	NUMERIC(7,0),
	ICUSTAY_ID	NUMERIC(7,0),
	ITEMID	NUMERIC(7,0),
	CHARTTIME	DATE,
	STORETIME	DATE,
	CGID	NUMERIC(7,0),
	VALUE	VARCHAR(200),
	VALUENUM	NUMERIC,
	VALUEUOM	VARCHAR(200),
	WARNING	NUMERIC(1,0),
	ERROR	NUMERIC(1,0),
	RESULTSTATUS	VARCHAR(200),
	STOPPED	VARCHAR(200)
);

COPY CHARTEVENTS FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/CHARTEVENTS.csv/CHARTEVENTS.csv' WITH (FORMAT csv, HEADER true);

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

COPY D_ITEMS FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/D_ITEMS.csv/D_ITEMS.csv' WITH (FORMAT csv, HEADER true);

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

COPY LABEVENTS FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/LABEVENTS.csv/LABEVENTS.csv' WITH (FORMAT csv, HEADER true);

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

COPY NOTEEVENTS FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/NOTEEVENTS.csv/NOTEEVENTS.csv' WITH (FORMAT csv, HEADER true);

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

COPY D_LABITEMS FROM '/Users/sangeethgeorge/MyProjects/oncoai-patient-outcome-navigator/data/mimic-iii-full/D_LABITEMS.csv/D_LABITEMS.csv' WITH (FORMAT csv, HEADER true);