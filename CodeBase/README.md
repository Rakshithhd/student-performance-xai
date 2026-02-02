## Why we take those Features : 
These features were selected because they represent academic history, engagement, and socio-educational context, all of which are known to influence final academic outcomes. 

## The Pass Target = 1 if G3 ≥ 10 else 0
Binary classification aligns better with intervention-based educational decision-making.

## Comparison Table
```
| Aspect         | Regression (G3)  | Classification (pass) |
| -------------- | ---------------- | --------------------- |
| Target         | G3 (numeric)     | pass (binary)         |
| Goal           | Grade prediction | Risk detection        |
| Practical use  | Analysis         | Intervention          |
| Complexity     | Higher noise     | More robust           |
| Explainability | Moderate         | Strong                |
```
We framed the problem in two ways: regression for academic analysis and classification for actionable intervention. Feature selection was guided by temporal availability, educational relevance, and explainability.


## df['pass'] = (df['G3'] >= 10).astype(int)
We converted the final grade into a binary pass/fail outcome to align the model with practical academic decision-making.

## df['attendance_ratio'] = df['absences'] / (df['absences'] + 1)
Attendance ratio provides a normalized representation of absenteeism that improves model robustness.
These transformations convert raw academic records into meaningful, interpretable variables that enhance prediction accuracy and align the model with real educational decision-making.

