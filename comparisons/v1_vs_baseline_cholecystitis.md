# Comparison: Evolved vs Baseline (cholecystitis, 10 common patients)

- **Baseline**: `/home/than/DeepLearning/mimic_skills/trajectories/baseline_cholecystitis_train10.json` (10 patients)
- **Evolved**: `/home/than/DeepLearning/mimic_skills/trajectories/v1_cholecystitis_train10.json` (10 patients)
- **Common patients**: 10

## Aggregate Metrics

| Metric | Baseline | Evolved | Delta |
|---|---|---|---|
| Diagnosis | 3/10 (30%) | 3/10 (30%) | +0% |
| Gracious Diagnosis | 3/10 (30%) | 3/10 (30%) | +0% |
| Physical Examination | 4/10 (40%) | 4/10 (40%) | +0% |
| Late Physical Examination | 5/10 (50%) | 4/10 (40%) | -10% |
| Action Parsing | 5/10 (50%) | 3/10 (30%) | -20% |
| Treatment Parsing | 4/10 (40%) | 3/10 (30%) | -10% |
| Diagnosis Parsing | 8/10 (80%) | 6/10 (60%) | -20% |
| Laboratory Tests | 1.10 | 0.80 | -0.30 |
| Imaging | 0.20 | 0.60 | +0.40 |
| Rounds | 1.70 | 1.10 | -0.60 |
| Invalid Tools | 4 | 1 | -3 |

## Per-Patient Diagnosis Changes

| hadm_id | Baseline Dx | Evolved Dx | Changed? |
|---|---|---|---|
| 20333063 | WRONG (Choledocholithiasis secondary ) | correct (Symptomatic Cholelithiasis wit) | FIXED |
| 20450633 | WRONG (Administer supportive care (fl) | WRONG (* Based on the overwhelming hi) | still wrong |
| 20690577 | correct (Acute Cholecystitis) | correct (Acute Cholecystitis) | ok |
| 21049206 | correct (Acute Cholecystitis) | WRONG (Inconclusive / Insufficient Da) | BROKEN |
| 21342222 | WRONG (Symptomatic Cholelithiasis / B) | WRONG (Symptomatic Cholelithiasis (Bi) | still wrong |
| 25376927 | WRONG (**Symptomatic Cholelithiasis**) | correct (Acute Cholecystitis) | FIXED |
| 26810924 | WRONG (* The combined clinical pictur) | WRONG (Acute viral hepatitis (HAV) | still wrong |
| 27309032 | WRONG (With mostly normal labs (excep) | WRONG (Symptomatic Cholelithiasis (li) | still wrong |
| 27635465 | WRONG (*) | WRONG (**Left-sided Obstructive Pyelo) | still wrong |
| 28733631 | correct (Acute Cholecystitis) | WRONG (* Acute Pancreatitis) | BROKEN |

**Fixed**: 2 patients | **Broken**: 2 patients | **Net**: +0

## Per-Patient PE Ordering Changes

| hadm_id | Baseline PE First | Evolved PE First | Changed? |
|---|---|---|---|
| 20333063 | yes | yes | ok |
| 20450633 | yes | NO | BROKEN |
| 20690577 | yes | NO | BROKEN |
| 21049206 | NO | NO | still wrong |
| 21342222 | NO | NO | still wrong |
| 25376927 | NO | yes | FIXED |
| 26810924 | NO | yes | FIXED |
| 27309032 | NO | yes | FIXED |
| 27635465 | NO | NO | still wrong |
| 28733631 | yes | NO | BROKEN |

**Fixed**: 3 patients | **Broken**: 3 patients | **Net**: +0

## Summary

- Diagnosis accuracy: 3/10 -> 3/10 (+0)
- PE first: 4/10 -> 4/10 (+0)
- Lab score total: 11 -> 8 (-3)
- Imaging score total: 2 -> 6 (+4)