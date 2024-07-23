SELECT distinct
    d.subject_id,
    MAX(CASE WHEN d.icd_code IN ('25000', 'E119', '25001', '25002', '25003', 'E109') THEN 1 ELSE 0 END) AS Diabetes,
    MAX(CASE WHEN d.icd_code IN ('40390', '5859', 'I129', 'N18.1', 'N18.2', 'N18.3', 'N18.4', 'N18.5', 'N18.6') THEN 1 ELSE 0 END) AS CKD
FROM `physionet-data.mimiciv_hosp.diagnoses_icd` d
GROUP BY d.subject_id;
 