# Gas Turbine Emission Prediction (CO & NOx)

This project implements machine learning models to predict flue gas emissions (Carbon Monoxide and Nitrogen Oxides) from a gas turbine based on ambient variables and sensor measurements.
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Viz-4C72B0)

![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-Environment-44A833?logo=anaconda&logoColor=white)

![Data Science](https://img.shields.io/badge/Data%20Science-Workflow-6A5ACD)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Production--Ready-FF6F00)

## 📊 Dataset Overview
The dataset contains 36,733 instances of 11 sensor variables collected from a gas turbine in Turkey. 
- **🔹 Input Features:** 

The model uses real operational parameters from a gas turbine system:

`AT` – Ambient Temperature

`AP` – Ambient Pressure

`AH` – Ambient Humidity

`AFDP` – Air Filter Difference Pressure

`GTEP` – Gas Turbine Exhaust Pressure

`TIT` – Turbine Inlet Temperature

`TAT`– Turbine After Temperature

`TEY` – Turbine Energy Yield

`CDP` – Compressor Discharge Pressure

These features capture environmental conditions, compressor behavior, and turbine thermodynamics, making them suitable for emissions prediction.

🎯 **Target Variables:**

The supervised learning task predicts regulated emission outputs:

`CO` – Carbon Monoxide emissions

`NOx` – Nitrogen Oxides emissions


## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Anaconda

## 📈 Methodology
1. **Temporal Splitting:** To mimic real-world scenarios, the data is split by years:
   - **Training:** 2011 - 2012
   - **Validation:** 2013
   - **Testing:** 2014 - 2015
2. **Feature Engineering:** 
   - Z-score Normalization (StandardScaler).
   - Correlation analysis showing a strong linear dependency between CDP, TEY, and GTEP.
   - Recursive Feature Elimination (RFE) to identify top predictors for each emission type.
3. **Modeling:** Comparison across multiple algorithms:
   - Linear, Ridge, and Lasso Regression
   - K-Nearest Neighbors (KNN)
   - Random Forest Regressor
   - Multi-Layer Perceptron (MLP) Neural Network

## 🚀 Key Results
- **Feature Importance:** Turbine parameters (TIT, CDP, TEY) show significantly higher correlation with emissions than ambient variables (AT, AP, AH).
- **Model Performance:** Ensemble methods (Random Forest) and Neural Networks (MLP) outperformed linear models, especially in capturing the non-linear behavior of NOx emissions.

## 📂 Repository Structure
- `gt_2011.csv` ... `gt_2015.csv`: Raw data files.
- `Gas_Turbine_Analysis.ipynb`: Main analysis and modeling notebook.
- `requirements.txt`: List of required Python packages.

## ⚙️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Ikennaphill/PEMS--Emission-detection-model.git
   
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
3. Run the app
   ```bash
   jupyter notebook 

📝 License
This project is open-source under the MIT License.
code
Code
---

### Part 3: Final Recommendations for Users
1.  **Data Notice:** If the `.csv` files are very large, GitHub might reject them. I recommend downloading the data (e.g., [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set)) directly from this source.
