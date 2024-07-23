with 
latest_labs AS (
    SELECT
        l.subject_id,
        l.itemid,
        l.valuenum,
        ROW_NUMBER() OVER (PARTITION BY l.subject_id, l.itemid ORDER BY l.charttime DESC) as rn
    FROM
        `physionet-data.mimiciv_hosp.labevents` l
    WHERE
        l.itemid IN (51221,50931,50912, 51006, 50862,50852)
        -- , 50907, 51000, 50904, 50852, 50905, 51082, 51069)
)
SELECT
    l.subject_id,
    MAX(CASE WHEN l.itemid = 51221 AND l.rn = 1 THEN l.valuenum END) AS Hematocrit_CKD,
    MAX(CASE WHEN l.itemid = 50931 AND l.rn = 1 THEN l.valuenum END) AS Glucose,
    MAX(CASE WHEN l.itemid = 50912 AND l.rn = 1 THEN l.valuenum END) AS Creatinine,
    MAX(CASE WHEN l.itemid = 51006 AND l.rn = 1 THEN l.valuenum END) AS Urea_Nitrogen,
    MAX(CASE WHEN l.itemid = 50862 AND l.rn = 1 THEN l.valuenum END) AS Albumin,
    MAX(CASE WHEN l.itemid = 50852 AND l.rn = 1 THEN l.valuenum END) AS HbA1c_Dia,
    -- MAX(CASE WHEN l.itemid = 50907 AND l.rn = 1 THEN l.valuenum END) AS Cholesterol_tot,
    -- MAX(CASE WHEN l.itemid = 51000 AND l.rn = 1 THEN l.valuenum END) AS Triglycerides,
    -- MAX(CASE WHEN l.itemid = 50904 AND l.rn = 1 THEN l.valuenum END) AS HDL,
    -- MAX(CASE WHEN l.itemid = 50905 AND l.rn = 1 THEN l.valuenum END) AS LDL,
    -- MAX(CASE WHEN l.itemid = 51082 AND l.rn = 1 THEN l.valuenum END) AS Creatinine_Ur,
    -- MAX(CASE WHEN l.itemid = 51069 AND l.rn = 1 THEN l.valuenum END) AS Albumin_Ur,
FROM
    latest_labs l
GROUP BY
    l.subject_id;
