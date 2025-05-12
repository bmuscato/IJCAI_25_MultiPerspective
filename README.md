# IJCAI_25_MultiPerspective


**Integrating Minority Voices: A Multi-Perspective Approach for more Inclusive Language Models**

**Abstract**

 In the realm of Natural Language Processing (NLP), common approaches for handling human disagreement include averaging annotators' viewpoints or using majority voting, both of which strive to establish a single ground truth. However, prior studies show that disregarding individual opinions can lead to the side-effect of under-representing minority perspectives, especially in subjective tasks, where annotators may systematically disagree because of their preferences. Recognizing that labels reflect the diverse backgrounds, life experiences, and values of individuals, this study proposes a new multi-perspective approach using soft labels to encourage the development of the next generation of perspective-aware models—more responsible, inclusive, and pluralistic. We conduct an extensive analysis across diverse subjective text classification tasks including  hate speech, irony, abusive language, and stance detection, to highlight the importance of capturing human disagreements, often overlooked by traditional aggregation methods.
Results show that the multi-perspective approach achieves superior classification performance (higher F1-scores), outperforming the traditional approaches that rely on a single ground-truth. Additionally, we assessed the models' ability to approximate human label distributions using Jensen-Shannon Divergence (JSD) as soft metric of reference. Notably, the multi-perspective models outperformed both baseline models across diverse subjective tasks. However, it exhibits lower model confidence in tasks like irony and stance detection, likely due to the inherent subjectivity present in the texts. Lastly, we exploit model uncertainty using Explainable AI (XAI) techniques and uncover meaningful insights into model predictions.


Paper Link


## 1.Prerequisites
requirements.txt includes the list of Python package requirements.


## 2. Data pre-processing 
Go to pre_processing folder 
data_preparation.ipynb includes the pre-processing functions.
gpt_summarization.ipynb contains instructions for document summarization.

## 3. Baseline 
Contains two baselines: majority vote, predicting hard label and ensemble, predicting first single annotations and then aggregating into one hard label.

## 4. MultiPerspective
Contains multi-perspective: input and outputs are represented as soft label, probability distribution. 

## 5. XAI
Contains feature-based attribution methods code for LIME, SHAP, Layer Integated Gradient (LIG). Attention matrices code and Layer Condunctance (LC). 

## 6. Appendix
Supplementary materials, including document summarization, fine-tuning details, and XAI visualizations is available here. 

