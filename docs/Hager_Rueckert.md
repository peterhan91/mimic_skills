# Evaluation and Mitigation of the Limitations of Large Language Models in Clinical Decision-Making

**Published in:** Nature Medicine | Volume 30 | September 2024 | Pages 2613–2622

**DOI:** [https://doi.org/10.1038/s41591-024-03097-1](https://doi.org/10.1038/s41591-024-03097-1)

**Received:** 26 January 2024 | **Accepted:** 29 May 2024 | **Published online:** 4 July 2024

---

## Authors

Paul Hager¹˒²˒⁸✉, Friederike Jungmann¹˒²˒⁸, Robbie Holland³, Kunal Bhagat⁴, Inga Hubrecht⁵, Manuel Knauer⁵, Jakob Vielhauer⁶, Marcus Makowski², Rickmer Braren²˒⁹, Georgios Kaissis¹˒²˒³˒⁷˒⁹ & Daniel Rueckert¹˒³˒⁹

### Affiliations

1. Institute for AI and Informatics, Klinikum rechts der Isar, Technical University of Munich, Munich, Germany
2. Institute for Diagnostic and Interventional Radiology, Klinikum rechts der Isar, Technical University of Munich, Munich, Germany
3. Department of Computing, Imperial College, London, UK
4. Department of Medicine, ChristianaCare Health System, Wilmington, DE, USA
5. Department of Medicine III, Klinikum rechts der Isar, Technical University of Munich, Munich, Germany
6. Department of Medicine II, University Hospital of the Ludwig Maximilian University of Munich, Munich, Germany
7. Reliable AI Group, Institute for Machine Learning in Biomedical Imaging, Helmholtz Munich, Munich, Germany
8. These authors contributed equally: Paul Hager, Friederike Jungmann
9. These authors jointly supervised this work: Rickmer Braren, Georgios Kaissis, Daniel Rueckert

✉ Correspondence: paul.hager@tum.de

---

## Abstract

Clinical decision-making is one of the most impactful parts of a physician's responsibilities and stands to benefit greatly from artificial intelligence solutions and large language models (LLMs) in particular. However, while LLMs have achieved excellent performance on medical licensing exams, these tests fail to assess many skills necessary for deployment in a realistic clinical decision-making environment, including gathering information, adhering to guidelines, and integrating into clinical workflows. Here we have created a curated dataset based on the Medical Information Mart for Intensive Care database spanning 2,400 real patient cases and four common abdominal pathologies as well as a framework to simulate a realistic clinical setting. We show that current state-of-the-art LLMs do not accurately diagnose patients across all pathologies (performing significantly worse than physicians), follow neither diagnostic nor treatment guidelines, and cannot interpret laboratory results, thus posing a serious risk to the health of patients. Furthermore, we move beyond diagnostic accuracy and demonstrate that they cannot be easily integrated into existing workflows because they often fail to follow instructions and are sensitive to both the quantity and order of information. Overall, our analysis reveals that LLMs are currently not ready for autonomous clinical decision-making while providing a dataset and framework to guide future studies.

---

## Introduction

Large language models (LLMs) have the potential to revolutionize our medical system, having shown their capabilities on diverse tasks. Importantly, as humans primarily interact with the world through language, LLMs are poised to be the point of access to the multimodal medical artificial intelligence (AI) solutions of the future. Until now, however, the diagnostic capabilities of models have been tested in structurally simple medical contexts, such as canonical vignettes of hypothetical patients or clinical case challenges. In both scenarios, all the required diagnostic information is provided upfront, and there is a single answer to be selected from a list of options. This type of question dominates both medical licensing exams, where LLMs score well above passing, and clinical case challenges, where models rival clinician performance.

However, while these medical licensing exams and clinical case challenges are suitable for testing the general medical knowledge of the test-taker, they are far removed from the daily and complex task of clinical decision-making. It is a multistep process that requires gathering and synthesizing data from diverse sources and continuously evaluating the facts to reach an evidence-based decision on a patient's diagnosis and treatment. As this process is very labor intensive, great potential exists in harnessing AI, such as LLMs, to alleviate much of the workload. LLMs can summarize reports, generate reports, serve as diagnostic assistants, and could ultimately autonomously diagnose patients. To understand how useful LLMs would be in such an autonomous, real-world setting, they must be evaluated on real-world data and under realistic conditions. However, the only analysis that tested an LLM throughout the diagnostic clinical workflow used curated lists of possible answers and examined only 36 hypothetical clinical vignettes. Furthermore, any model that is used in such a high-stakes clinical context must not only be highly accurate, but also adhere to diagnostic and treatment guidelines, be robust, and follow instructions, all of which have not been tested in previous medical evaluations.

Here, we present a curated dataset based on the Medical Information Mart for Intensive Care (MIMIC-IV) database spanning 2,400 real patient cases and 4 common abdominal pathologies (appendicitis, pancreatitis, cholecystitis and diverticulitis) as well as a comprehensive evaluation framework around our dataset to simulate a realistic clinical setting. We provide LLMs with a patient's history of present illness and ask them to iteratively gather and synthesize additional information such as physical examinations, laboratory results and imaging reports until they are confident enough to provide a diagnosis and treatment plan. Our dataset, task and analysis comprise a large-scale evaluation of LLMs on everyday clinical decision-making tasks in a realistic, open-ended environment. Unlike previous works, we test the autonomous information-gathering and open-ended diagnostic capabilities of models, representing an essential step toward evaluating their suitability as clinical decision-makers.

To understand how useful LLMs would be as second readers, we compare the diagnostic accuracy of the models with that of clinicians. Furthermore, we propose and evaluate a range of characteristics beyond diagnostic accuracy, such as adherence to diagnostic and treatment guidelines, correct interpretation of laboratory test results, instruction-following capabilities, and robustness to changes in instructions, information order and information quantity. Finally, we show that summarizing progress and filtering laboratory results for only abnormal results addresses some of the current limitations of models. We make our evaluation framework and dataset freely and openly available to guide future studies considering the use of LLMs in clinical practice.

---

## Results

### Creating the MIMIC-CDM Dataset and Evaluation Framework

Our curated dataset, MIMIC-IV-Ext Clinical Decision Making (MIMIC-CDM), is created using the well-established MIMIC-IV database, which contains de-identified electronic health records. Our dataset contains data from 2,400 unique patients presenting with acute abdominal pain to the emergency department and whose primary diagnosis was one of the following pathologies: appendicitis, cholecystitis, diverticulitis or pancreatitis. We chose these target pathologies as they represent clinically important diagnoses of a common chief complaint, abdominal pain, which accounts for 10% of all emergency department visits. Importantly, good differentiation between the four pathologies can be achieved using standard diagnostic tests, all of which are present in our dataset.

**MIMIC-CDM Statistics (2,400 cases):**

| Pathology | Cases | Imaging Modality | Count |
|---|---|---|---|
| Appendicitis | 957 | CT abdomen | 1,836 |
| Cholecystitis | 648 | Radiograph chest | 1,728 |
| Diverticulitis | 257 | US abdomen | 1,325 |
| Pancreatitis | 538 | Radiograph abdomen | 342 |
| | | MRCP abdomen | 227 |
| | | Other | 501 |

- 2,400 physical examinations
- 143,191 laboratory tests
- 5,959 radiologist reports

To reflect a realistic clinical setting that allows LLMs to autonomously engage in every step of the clinical decision-making process, we have created a comprehensive evaluation framework around our dataset. For comparisons with practicing clinicians and further tests concerning robustness, we evaluate the diagnostic accuracy of LLMs as second readers, providing all necessary information for a diagnosis upfront, which we call MIMIC-IV-Ext Clinical Decision Making with Full Information (MIMIC-CDM-FI).

**Models tested:** Llama 2 Chat (70B), Open Assistant (OASST) (70B), WizardLM (70B), Clinical Camel (70B), and Meditron (70B). Due to MIMIC data privacy restrictions, neither ChatGPT, GPT-4, nor Med-PaLM could be tested.

### LLMs Diagnose Significantly Worse Than Clinicians

We compared the diagnostic accuracy of the models on a subset of 80 patients of MIMIC-CDM-FI to four hospitalists with varying degrees of experience and from two countries.

Current LLMs perform significantly worse than clinicians on aggregate across all diseases (P < 0.001 for all model–doctor comparisons). The difference in mean diagnostic performance between doctors and models was large, ranging from 16 to 25 points. The diagnostic accuracy between the clinicians varied, with the German hospitalists in residency (mean = 87.50% ± 3.68%) performing slightly worse than the more senior US hospitalist (mean = 92.50%).

**Diagnostic Accuracy (%) — Full Information (MIMIC-CDM-FI), subset n=80:**

| Pathology | Llama 2 Chat | OASST | WizardLM | Clinical Camel | Meditron | Doctors |
|---|---|---|---|---|---|---|
| Appendicitis | 89 | 100 | 100 | 100 | 99 | 96 |
| Cholecystitis | 62 | 63 | 46 | 68 | 13 | 84 |
| Diverticulitis | 55 | 46 | 50 | 59 | 35 | 79 |
| Pancreatitis | 58 | 74 | 76 | 84 | 89 | 86 |
| Mean | 66 | 69 | 66 | 73 | 64 | 89 |

Most models were able to match clinician performance on the simplest diagnosis, appendicitis, where 3 of 4 clinicians also correctly diagnosed 20 of 20 patients. Neither of the two specialist models performed significantly better on aggregate.

**Diagnostic Accuracy (%) — Autonomous Clinical Decision-Making (MIMIC-CDM):**

| Pathology | Llama 2 Chat | OASST | WizardLM |
|---|---|---|---|
| Appendicitis | 74.6 | 82.0 | 78.4 |
| Cholecystitis | 33.8 | 48.0 | 47.4 |
| Diverticulitis | 34.2 | 45.5 | 44.4 |
| Pancreatitis | 39.2 | 44.1 | 45.5 |
| Mean | 45.5 | 54.9 | 53.9 |

In the autonomous scenario, a general decrease in performance was observed compared to MIMIC-CDM-FI across all pathologies.

### Current LLMs Are Hasty and Unsafe Clinical Decision-Makers

In addition to poor diagnostic accuracy, LLMs often fail to order the exams required by diagnostic guidelines, do not follow treatment guidelines, and are incapable of interpreting lab results.

**Physical Examinations:**
- Llama 2 Chat: 97.1% requested as first action; 98.1% requested at all
- OASST: 79.8% first; 87.7% at all
- WizardLM: 53.1% first; 63.9% at all

**Laboratory Test Interpretation:** All LLMs performed very poorly, especially in the critical categories:
- Low test results: Chat 26.5%, OASST 70.2%, WizardLM 45.8%
- Normal test results: Chat 89.7%, OASST 90.7%, WizardLM 93.2%
- High test results: Chat 50.1%, OASST 77.2%, WizardLM 24.1%

**Treatment Recommendations:** LLMs consistently did not recommend appropriate and sufficient treatment, especially for patients with more severe forms of pathologies. While consistent in recommending appendectomy for appendicitis (~97%) and antibiotics for diverticulitis (~86-97%), they rarely recommended other treatments when appropriate such as colectomy for perforated colons or drainage of infected pancreatic necrosis.

### Current LLMs Require Extensive Clinician Supervision

**Instruction Following:** All models struggle to follow provided instructions, making errors every two to four patients when providing actions and hallucinating nonexistent tools every two to five patients.

**Sensitivity to Instruction Phrasing:** Changes in instructions can lead to large changes in diagnostic accuracy. For example:
- Removing system and user instructions: up to +5.1% for Chat on cholecystitis, down to −16.0% for Chat on pancreatitis
- Asking for "main diagnosis" instead of "final diagnosis": up to +7.0% for Chat on diverticulitis, down to −10.6% for WizardLM on cholecystitis

**Sensitivity to Information Quantity:** Models perform worse when all diagnostic exams are provided, typically attaining best performance when only a single exam is provided. Removing information greatly increases diagnostic accuracy:
- Cholecystitis: +18.5% (Chat) and +16.5% (WizardLM) with only radiologist reports
- Pancreatitis: +21.6% (Chat), +9.5% (OASST), +8.6% (WizardLM) with only laboratory results

**Sensitivity to Information Order:** Changing the order of diagnostic information changes diagnostic accuracy:
- Up to 18.0% range for Chat on pancreatitis
- Up to 7.9% range for OASST on cholecystitis
- Up to 5.4% range for WizardLM on cholecystitis

The order delivering best performance differs for each model and each pathology.

### First Steps Toward Mitigating Limitations of Current LLMs

- **Automatic summarization protocol:** Removing summarization resulted in marginal but consistent losses: −1.3% (Chat), −0.8% (OASST), −0.5% (WizardLM) on mean, particularly hurting diverticulitis diagnosis.
- **Filtering for abnormal laboratory results:** Removing all normal test results generally improved performance on the MIMIC-CDM-FI dataset.

---

## Discussion

The strong performance of LLMs on medical licensing exams has led to increased interest in using them in clinical decision-making scenarios involving real patients. However, medical licensing exams do not test the capabilities required for real-world clinical decision-making. This study evaluated leading open-access LLMs in an autonomous clinical decision-making scenario with thousands of real-world cases.

### Key Findings

1. **No model consistently reached physician-level diagnostic accuracy** across all pathologies, with further decrease when gathering information autonomously.
2. **LLMs cannot classify lab results as normal or abnormal**, even when provided with reference ranges.
3. **Models do not follow diagnostic guidelines**, indicating a tendency to diagnose before fully understanding a patient's case.
4. **Treatment recommendations are insufficient**, especially for severe cases where emergency operations were indicated.
5. **Models are sensitive to instruction phrasing**, information quantity, and information order — all in disease-specific ways.
6. **Instruction-following is unreliable**, with frequent hallucination of nonexistent tools.

### Limitations

- Dataset limited to four abdominal pathologies; a fully autonomous model must handle all possible diagnoses.
- Clear bias toward USA (English data, American guidelines, predominantly English training data).
- Only open-access Llama 2-based models could be tested due to MIMIC data privacy restrictions.
- Limited reader study (3 German residents + 1 senior US hospitalist).

### Conclusion

Current models do not achieve satisfactory diagnostic accuracy, performing significantly worse than trained physicians, and do not follow treatment guidelines, posing a serious risk to patient health. LLMs are distracted by relevant diagnostic information, are sensitive to the order of diagnostic tests, and struggle to follow instructions, prohibiting autonomous deployment and requiring extensive clinician supervision.

---

## Methods

### MIMIC-CDM Dataset

Created from the MIMIC-IV Database containing de-identified electronic health records of ~300,000 patients from Beth Israel Deaconess Medical Center in Boston, MA, USA (2008–2019).

**Data Processing Steps:**
1. Filtered patients by ICD diagnosis codes for target pathologies
2. Manually verified discharge diagnosis for each patient
3. Removed patients with pathology mentioned in history of present illness
4. Removed patients without physical examinations
5. Gathered laboratory/microbiology tests (including up to 1 day before admission)
6. Collected radiology reports with anatomical region and modality classification
7. Extracted procedures and operations from ICD codes and discharge summaries
8. Replaced remaining pathology mentions with censoring markers
9. Excluded patients without laboratory tests or abdominal imaging

**Final Dataset:** 2,400 patients (957 appendicitis, 648 cholecystitis, 257 diverticulitis, 538 pancreatitis) with 138,788 laboratory results from 480 unique tests, 4,403 microbiology results, and 5,959 radiology reports.

### Reader Study

- 4 hospitalists: 2 from TU Munich (2–3 years experience), 1 from LMU Munich (4 years), 1 from Christiana Care, Delaware, USA (29 years)
- Subset of 80 patients (20 per pathology) + 5 distractor cases
- Each LLM evaluated 20 times with different random seeds over the 80-patient subset

### Evaluation Framework

LLMs are presented with a patient's history of present illness and tasked to iteratively gather information using available tools:
- **Physical examination**
- **Laboratory test** (with specific test names)
- **Imaging** (with modality and region)

Models use a "think-act" format: consider evidence, then either request more information or provide a final diagnosis and treatment plan. Context length limit: 4,096 tokens. Automatic summarization protocol used when approaching limits.

### Models

All models based on Llama 2 (70B) with GPTQ quantization (4-bit, 32 group size). Selected for:
- Context length ≥ 4,096 tokens
- Strong MedQA (USMLE) performance
- Open-access/downloadable (required by MIMIC data usage agreement)

| Model | Base | Parameters | Downloadable |
|---|---|---|---|
| Llama 2 Chat | Llama 2 | 70B | ✓ |
| OASST | Llama 2 | 70B | ✓ |
| WizardLM | Llama 2 | 70B | ✓ |
| Clinical Camel | Llama 2 | 70B | ✓ |
| Meditron | Llama 2 | 70B | ✓ |
| Chat-GPT | GPT3.5 | ??? | ✗ |
| GPT-4 | ??? | ??? | ✗ |
| Med-PaLM | Flan-PaLM | 540B | ✗ |
| Med-PaLM 2 | PaLM 2 | 340B | ✗ |

### Metrics and Statistical Analysis

- Per-class accuracy used throughout (number of correct diagnoses / total patients with that disease)
- Two-sided Student's t-tests with unequal variances
- Bonferroni correction: multiplier of 5 (doctors vs models), 3 (specialist vs generalist)
- Python 3.10, SciPy library

---

## Data Availability

The dataset is available to researchers with MIMIC-IV access:
- PhysioNet account: [https://physionet.org/](https://physionet.org/)
- MIMIC-IV database: [https://physionet.org/content/mimiciv/2.2/](https://physionet.org/content/mimiciv/2.2/)
- Dataset creation code: [https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset/](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset/)
- Direct download: [https://www.physionet.org/content/mimic-iv-ext-cdm/1.0/](https://www.physionet.org/content/mimic-iv-ext-cdm/1.0/)

## Code Availability

- Evaluation framework: [https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework/](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Framework/)
- Analysis framework: [https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Analysis/](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Analysis/)

**Dependencies:** Python v3.10, pytorch v2.1.1, transformers v4.35.2, spacy v3.4.4, langchain v0.0.339, optimum v1.14, thefuzz v0.20, exllamav2 v0.0.8, nltk v3.8.1, negspacy v1.0.4, scispacy v0.5.2

---

## References

1. Thirunavukarasu, A. J. et al. Large language models in medicine. *Nat. Med.* **29**, 1930–1940 (2023).
2. Lee, S. et al. LLM-CXR: instruction-finetuned LLM for CXR image understanding and generation. In *12th International Conference on Learning Representations* (ICLR, 2024).
3. Van Veen, D. et al. RadAdapt: radiology report summarization via lightweight domain adaptation of large language models. In *Proc. 22nd Workshop on Biomedical Natural Language Processing and BioNLP Shared Tasks* 449–460 (Association for Computational Linguistics, 2023).
4. Tu, T. et al. Towards generalist biomedical AI. *NEJM AI* **1**, AIoa2300138 (2024).
5. Van Veen, D. et al. Adapted large language models can outperform medical experts in clinical text summarization. *Nat. Med.* **30**, 1134–1142 (2024).
6. van Sonsbeek, T. et al. Open-ended medical visual question answering through prefix tuning of language models. In *Proc. MICCAI 2023* 726–736 (Springer, 2023).
7. Bazi, Y., Rahhal, M. M. A., Bashmal, L. & Zuair, M. Vision–language model for visual question answering in medical imagery. *Bioengineering* **10**, 380 (2023).
8. Singhal, K. et al. Towards expert-level medical question answering with large language models. Preprint at https://arxiv.org/abs/2305.09617 (2023).
9. Singhal, K. et al. Large language models encode clinical knowledge. *Nature* **620**, 172–180 (2023).
10. Nori, H., King, N., McKinney, S. M., Carignan, D. & Horvitz, E. Capabilities of GPT-4 on medical challenge problems. Preprint at https://arxiv.org/abs/2303.13375 (2023).
11. Belyaeva, A. et al. Multimodal llms for health grounded in individual-specific data. In *Proc. Workshop on Machine Learning for Multimodal Healthcare Data* 86–102 (Springer, 2023).
12. Moor, M. et al. Foundation models for generalist medical artificial intelligence. *Nature* **616**, 259–265 (2023).
13. Jin, D. et al. What disease does this patient have? A large-scale open domain question answering dataset from medical exams. *Appl. Sci.* **11**, 6421 (2021).
14. Hendrycks, D. et al. Measuring massive multitask language understanding. In *Proc. ICLR* (2020).
15. Thirunavukarasu, A. J. et al. Trialling a large language model (chatGPT) in general practice with the applied knowledge test. *JMIR Med. Educ.* **9**, 46599 (2023).
16. Pal, A., Umapathi, L. K. & Sankarasubbu, M. MedMCQA: a large-scale multi-subject multi-choice dataset for medical domain question answering. In *Proc. Conference on Health, Inference, and Learning* 248–260 (PMLR, 2022).
17. Kung, T. H. et al. Performance of chatgpt on usmle: potential for ai-assisted medical education using large language models. *PLoS Digital Health* **2**, 0000198 (2023).
18. Gilson, A. et al. How does chatgpt perform on the United States medical licensing examination? *JMIR Med. Educ.* **9**, 45312 (2023).
19. Toma, A. et al. Clinical camel: an open-source expert-level medical language model with dialogue-based knowledge encoding. Preprint at https://arxiv.org/abs/2305.12031 (2023).
20. Nori, H. et al. Can generalist foundation models outcompete special-purpose tuning? Case study in medicine. Preprint at https://arxiv.org/abs/2311.16452 (2023).
21. McDuf, D. et al. Towards accurate differential diagnosis with large language models. Preprint at https://arxiv.org/abs/2312.00164 (2023).
22. Kanjee, Z., Crowe, B. & Rodman, A. Accuracy of a generative artificial intelligence model in a complex diagnostic challenge. *JAMA* **3**, 78–80 (2023).
23. Buckley, T., Diao, J. A., Rodman, A. & Manrai, A.K. Accuracy of a vision-language model on challenging medical cases. Preprint at https://arxiv.org/abs/2311.05591 (2023).
24. Eriksen, A. V., Möller, S. & Ryg, J. Use of GPT-4 to diagnose complex clinical cases. *NEJM AI* **1**, AIp2300031 (2023).
25. Berman, S. in *Berman's Pediatric Decision Making* 5th edn 1–6 (Mosby, 2011).
26. Tiffen, J., Corbridge, S. J. & Slimmer, L. Enhancing clinical decision making: development of a contiguous definition and conceptual framework. *J. Prof. Nursing* **30**, 399–405 (2014).
27. Shortliffe, E. H. & Sepúlveda, M. J. Clinical decision support in the era of artificial intelligence. *JAMA* **320**, 2199–2200 (2018).
28. Rao, A. et al. Assessing the utility of chatgpt throughout the entire clinical workflow. *J. Med. Int. Res.* **25**, 48659 (2023).
29. Goldberger, A. L. et al. Physiobank, physiotoolkit, and physionet. *Circulation* **101**, 215–220 (2000).
30. Cervellin, G. et al. Epidemiology and outcomes of acute abdominal pain in a large urban emergency department. *Ann. Transl. Med.* (2016).
31. Di Saverio, S. et al. Diagnosis and treatment of acute appendicitis: 2020 update of the WSES Jerusalem guidelines. *World J. Emerg. Surg.* (2020).
32. Touvron, H. et al. Llama 2: open foundation and fine-tuned chat models. Preprint at https://arxiv.org/abs/2307.09288 (2023).
33. Köpf, A. et al. OpenAssistant conversations—democratizing large language model alignment. In *NeurIPS* 47669–47681 (2024).
34. Xu, C. et al. Wizardlm: Empowering large language models to follow complex instructions. In *ICLR* (2024).
35. Chen, Z. et al. MEDITRON-70B: scaling medical pretraining for large language models. Preprint at https://arxiv.org/abs/2311.16079 (2023).
36. Di Saverio, S. et al. Diagnosis and treatment of acute appendicitis: 2020 update of the WSES Jerusalem guidelines. *World J. Emerg. Surg.* **15**, 27 (2020).
37. Pisano, M. et al. 2020 WSES updated guidelines for the diagnosis and treatment of acute calculus cholecystitis. *World J. Emerg. Surg.* **15**, 61 (2020).
38. Hall, J. et al. The american society of colon and rectal surgeons clinical practice guidelines for the treatment of left-sided colonic diverticulitis. *Dis. Colon Rectum* **63**, 728–747 (2020).
39. Leppäniemi, A. et al. 2019 WSES guidelines for the management of severe acute pancreatitis. *World J. Emerg. Surg.* **14**, 27 (2019).
40. Dettmers, T., Pagnoni, A., Holtzman, A. & Zettlemoyer, L. QLoRA: efficient finetuning of quantized LLMs. In *NeurIPS* 10088–10115 (2024).
41. Lester, B., Al-Rfou, R. & Constant, N. The power of scale for parameter-efficient prompt tuning. In *Proc. EMNLP* 3045–3059 (2021).
42. Guo, Q. et al. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. In *ICLR* (2024).
43. Shi, F. et al. Large language models can be easily distracted by irrelevant context. In *Proc. ICML* 31210–31227 (PMLR, 2023).
44. Yang, C. et al. Large language models as optimizers. In *ICLR* (2023).
45. Zheng, C., Zhou, H., Meng, F., Zhou, J. & Huang, M. On large language models' selection bias in multi-choice questions. In *ICLR* (2024).
46. Pezeshkpour, P. & Hruschka, E. Large language models sensitivity to the order of options in multiple-choice questions. Preprint at https://arxiv.org/abs/2308.11483 (2023).
47. Liu, N. F. et al. Lost in the middle: how language models use long contexts. *Trans. Assoc. Comput. Linguist.* **12**, 157–173 (2024).
48. Testolin, A. Can neural networks do arithmetic? *Appl. Sci.* **14**, 744 (2024).
49. Dziri, N. et al. Faith and fate: limits of transformers on compositionality. In *NeurIPS* 70293–70332 (2024).
50. Golkar, S. et al. xVal: a continuous number encoding for large language models. In *NeurIPS 2023 AI for Science Workshop* (2023).
51. Thawani, A., Pujara, J., Ilievski, F. & Szekely, P. Representing numbers in NLP: a survey and a vision. In *Proc. NAACL-HLT* 644–656 (2021).
52. Zhou, J. et al. Instruction-following evaluation for large language models. Preprint at https://arxiv.org/abs/2311.07911 (2023).
53. Cummings, M. L. in *Decision Making in Aviation* 289–294 (Routledge, 2017).
54. Lyell, D. & Coiera, E. Automation bias and verification complexity: a systematic review. *J. Am. Med. Inform. Assoc.* **24**, 423–431 (2017).
55. Tschandl, P. et al. Human–computer collaboration for skin cancer recognition. *Nat. Med.* (2020).
56. Kiani, A. et al. Impact of a deep learning assistant on the histopathologic classification of liver cancer. *NPJ Digit. Med.* **3**, 23 (2020).
57. DeCamp, M. & Lindvall, C. Mitigating bias in AI at the point of care. *Science* **381**, 150–152 (2023).
58. Together Computer. RedPajama: an open dataset for training large language models. GitHub (2023).
59. Ouyang, L. et al. Training language models to follow instructions with human feedback. In *NeurIPS* 27730–27744 (2022).
60. Brown, T. et al. Language models are few-shot learners. In *NeurIPS* 1877–1901 (2020).
61. Roberts, A. et al. Exploring the limits of transfer learning with a unified text-to-text transformer. *J. Mach. Learn. Res.* **21**, 1–67 (2020).
62. Radford, A. et al. Language models are unsupervised multitask learners. *OpenAI blog* **1**, 9 (2019).
63. Kaplan, J. et al. Scaling laws for neural language models. Preprint at https://arxiv.org/abs/2001.08361 (2020).
64. OpenAI. GPT-4 technical report. Preprint at https://arxiv.org/abs/2303.08774 (2023).
65. Chung, H. W. et al. Scaling instruction-finetuned language models. *J. Mach. Learn. Res.* **25**, 1–53 (2024).
66. Abacha, A. B. et al. Bridging the gap between consumers' medication questions and trusted answers. *Stud. Health Technol. Inform.* **264**, 25–29 (2019).
67. Abacha, A. B., Agichtein, E., Pinter, Y. & Demner-Fushman, D. Overview of the medical question answering task at TREC 2017 LiveQA (2017).
68. Anil, R. et al. PaLM 2 technical report. Preprint at https://arxiv.org/abs/2305.10403 (2023).
69. Wang, Y. & Zhao, Y. TRAM: benchmarking temporal reasoning for large language models. Preprint at https://arxiv.org/abs/2310.00835 (2023).
70. McKinney, S. M. et al. International evaluation of an AI system for breast cancer screening. *Nature* **577**, 89–94 (2020).
71. Wei, J. et al. Chain-of-thought prompting elicits reasoning in large language models. In *NeurIPS* 24824–24837 (2022).
72. Yao, S. et al. ReAct: synergizing reasoning and acting in language models. In *ICLR* (2023).
73. ML for Computational Physiology. Responsible use of MIMIC data with online services like GPT. PhysioNet (2023).
74. Toma, A. et al. Generative AI could revolutionize health care—but not if control is ceded to big tech. *Nature* **624**, 36–38 (2023).
75. Frantar, E., Ashkboos, S., Hoefler, T. & Alistarh, D. GPTQ: accurate post-training quantization for generative pre-trained transformers. In *ICLR* (2023).
76. Haibe-Kains, B. et al. Transparency and reproducibility in artificial intelligence. *Nature* **586**, 14–16 (2020).

---

## Acknowledgements

D.R. received funding via the European Research Council Grant Deep4MI (884622). Open access funding provided by Technische Universität München.

## Author Contributions

P.H. and F.J. conceptualized the study. P.H. wrote the code for dataset creation, model execution, and evaluation. F.J. made all medical decisions. K.B., I.H., M.K., and J.V. participated in the reader study. D.R., G.K., and R.B. supervised the project. R.H. assisted with coding design and paper structure.

## Competing Interests

The authors declare no competing interests.

---

**License:** This article is licensed under a Creative Commons Attribution 4.0 International License.

© The Author(s) 2024
