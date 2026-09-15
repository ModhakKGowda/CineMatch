# 🎬 CineMatch: Movie Recommendation System

CineMatch is an intelligent recommendation engine built on the MovieLens dataset. It addresses user "Analysis Paralysis" by delivering personalized movie recommendations using Collaborative Filtering and Matrix Factorization.

## 📌 Project Architecture & Approach

1. **Exploratory Data Analysis (`eda.py`)**
   * Plotted long-tail distribution of movie popularities.
   * Calculated Utility Matrix sparsity ratio (~98.3%).

2. **Memory-Based Collaborative Filtering (`recommender.py`)**
   * **User-User Filtering:** Computes Cosine Similarity across user vectors to predict ratings.
   * **Item-Item Filtering:** Analyzes item-item similarity matrices to recommend similar titles.

3. **Model-Based Collaborative Filtering (SVD)**
   * Utilizes Singular Value Decomposition via `surprise` to compress sparse matrix representations into latent vector embeddings.

4. **Web Interface (`app.py`)**
   * Flask REST API delivering live predictions to a clean web dashboard.

---

## 📊 Performance Benchmark & Evaluation

| Algorithm | Model Type | Evaluation Metric (RMSE) | Performance |
| :--- | :--- | :--- | :--- |
| **User-Based CF** | Cosine Similarity | ~0.992 | Baseline |
| **Item-Based CF** | Cosine Similarity | ~0.974 | Moderate |
| **SVD Matrix Factorization** | Latent Factors ($k=50$) | **~0.873** | **Optimal** |

---

## 🚀 How to Run locally / in Codespaces

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Data Analysis
python eda.py

# 3. Test Recommender Pipeline
python recommender.py

# 4. Launch Flask Web Application
python app.py
