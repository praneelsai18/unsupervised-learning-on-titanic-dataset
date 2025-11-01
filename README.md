# Unsupervised Learning on Titanic Dataset (Stand-Alone)

## 🧠 Overview
This project demonstrates the use of **Unsupervised Machine Learning** techniques on a standalone Titanic-like dataset.  
It includes:

✅ **K-Means Clustering**  
✅ **PCA Dimensionality Reduction**  
✅ **Cluster Visualization**  
✅ No dependency on any previous project  

---

## 🎯 Objectives

### ✅ 1. Data Preprocessing
- A Titanic dataset is generated inside the script.
- Categorical features (`Sex`, `Embarked`) are numerically encoded.
- All features are standardized using `StandardScaler`.

### ✅ 2. Clustering (K-Means)
- Apply **K-Means** with 2 clusters.
- Assign cluster labels to each passenger.

### ✅ 3. Dimensionality Reduction (PCA)
- Reduce dataset to **2 principal components**.
- Visualize the clusters in 2D space.

---

## 📁 Project Structure

mini_project_5_titanic_unsupervised/
│ ├── main.py
├── requirements.txt 
└── plots/
└── titanic_clusters.png   # PCA cluster visualization

---

## 🛠️ Installation

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt

Step 2 — Run the project

python main.py


---

📊 Sample Output

✅ Titanic Dataset Created Successfully!
✅ Data Standardized!
✅ K-Means Clustering Completed!
✅ PCA Completed!
📊 Cluster plot saved to /plots/titanic_clusters.png
✅ Mini Project 5 Completed Successfully!


---

📈 Visualization

A scatter plot is created using PCA components:

Each point = passenger

Colors represent K-means clusters

Easy visual understanding of cluster separation


Saved at:

plots/titanic_clusters.png


---

✅ Conclusion

In this project, unsupervised learning was successfully applied to the Titanic dataset.

K-Means helped identify hidden patterns in the passenger data.

PCA reduced the dataset to 2 dimensions, enabling easy visualization.

This demonstrates the fundamental workflow of clustering + dimensionality reduction in machine learning.



---

👨‍💻 Author

G. Sai Praneel
Department of Computer Science & Engineering
