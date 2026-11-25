========================================================================
README: Multiclass Social Media Dataset for Mental Health and Emotion Classification
========================================================================

1. DATASET OVERVIEW
-------------------
- File Name: Mental_Health_6Class_Final_Cleaned.csv
- Total Records: 42,518
- Features/Columns: Unnamed: 0 (Index), Text, Emotion
- Condition: Highly cleaned, filtered, and checked for zero duplication.

2. CLASS DISTRIBUTION & MENTAL HEALTH INDICATORS
------------------------------------------------
The dataset is structured into 6 distinct classes, capturing both standard emotions and critical mental health states:

- Positive: 9,766 records
- Neutral: 7,910 records
- Anger: 7,121 records
- Depression: 6,762 records
- Sadness: 6,221 records
- Anxiety: 4,738 records

3. COLUMN DESCRIPTIONS
-----------------------
- Unnamed: 0: A unique sequence or index identifier for each row.
- Text: Cleaned textual sentences capturing user experiences, thoughts, and emotional expressions.
- Emotion: The target categorical label (Positive, Neutral, Anger, Depression, Sadness, Anxiety).

4. SPECIFIC ADVANTAGES OF THIS DATASET
--------------------------------------
- No Data Leakage: Complete deduplication ensures that cross-validation and evaluation metrics remain unbiased during model training.
- Balanced Structure: The distribution across all 6 classes is relatively balanced, minimizing model bias towards any single majority class.
- Multi-Class Mental Health Focus: Combines general sentiment (Positive/Neutral) with high-risk psychological states (Depression/Anxiety/Anger/Sadness).

5. USAGE & CITATION
-------------------
If you use this dataset in your research, please cite the Mendeley Data DOI provided on the main repository page.
========================================================================