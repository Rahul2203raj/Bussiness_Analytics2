Medical Cost Prediction project
Target: annual_medical_cost
Model: Linear Regression
8 inputs: previous_year_cost, age, bmi, smoker, hospital_admissions, doctor_visits_per_year, medication_count, insurance_coverage_pct
Split: 80/20; random_state=42.
Run notebook to regenerate model. App requires medical_cost_model.sav alongside app.py.

                       Metric                                                                                                                       Value
                       Target                                                                                                         annual_medical_cost
                        Model                                                                                                           Linear Regression
      Selected input features previous_year_cost, age, bmi, smoker, hospital_admissions, doctor_visits_per_year, medication_count, insurance_coverage_pct
         Rows before cleaning                                                                                                                        5000
          Rows after cleaning                                                                                                                        5000
                      Columns                                                                                                                          20
Missing cells before cleaning                                                                                                                        1048
 Missing cells after cleaning                                                                                                                           0
       Duplicate rows removed                                                                                                                           0
          Training rows (80%)                                                                                                                        4000
           Testing rows (20%)                                                                                                                        1000
                           R²                                                                                                                    0.891537
                          MAE                                                                                                                 1578.085422
                         RMSE                                                                                                                  2300.10058
                     MAPE (%)                                                                                                                   34.209686