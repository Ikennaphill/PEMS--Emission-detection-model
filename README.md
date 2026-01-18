# Gas Turbine Emission Prediction (CO & NOx)

This project implements machine learning models to predict flue gas emissions (Carbon Monoxide and Nitrogen Oxides) from a gas turbine based on ambient variables and sensor measurements.

## 📊 Dataset Overview
The dataset contains 36,733 instances of 11 sensor variables collected from a gas turbine in Turkey. 
- **Features:** Ambient Temperature (AT), Ambient Pressure (AP), Ambient Humidity (AH), Air Filter Difference Pressure (AFDP), Gas Turbine Exhaust Pressure (GTEP), Turbine Inlet Temperature (TIT), Turbine After Temperature (TAT), Turbine Energy Yield (TEY), Compressor Discharge Pressure (CDP).
- **Targets:** Carbon Monoxide (CO), Nitrogen Oxides (NOx).

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
   git clone https://github.com/yourusername/gas-turbine-prediction.git

2. Install dependencies:
pip install -r requirements.txt

3. Run the Jupyter Notebook:
jupyter notebook

📝 License
This project is open-source under the MIT License.
code
Code
---

### Part 3: Final Recommendations for GitHub
1.  **Data Notice:** If the `.csv` files are very large, GitHub might reject them. I recommend mentioning in the README where users can download the data (e.g., [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set)).
2.  **Add a `.gitignore`:** Create a file named `.gitignore` and add `.ipynb_checkpoints/`, `__pycache__/`, and `.DS_Store` to keep your repo clean.
3.  **Requirements File:** Create a `requirements.txt` containing:
    ```text
    numpy
    pandas
    seaborn
    matplotlib
    scikit-learn
    ```
4.  **Visuals:** Before you take your final screenshots/commits, ensure your plots have titles and axis labels. Your heatmap is already excellent—it will be a great "hero image" for your project.
