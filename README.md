
# Context-Aware Bangla Homograph Disambiguation

## Overview

This project addresses the challenge of disambiguating homographs in Bangla, where identically spelled words may have different meanings or pronunciations based on context. Using machine learning techniques, we have developed and compared several models, including Naive Bayes, SVM, Logistic Regression, Random Forest, and XGBoost, to accurately classify homographs in context.

The study focuses on five homograph pairs:
- **Dak_Dako** (ডাক/ডাকো)
- **Bol_Bolo** (বল/বলো)
- **Kal_Kalo** (কাল/কালো)
- **Komol_Komlo** (কমল/কমলো)
- **Mot_Moto** (মত/মতো)

Our work highlights the importance of feature engineering, hyperparameter tuning, and robust model evaluation to improve accuracy in Bangla Text-to-Speech (TTS) synthesis systems.

## Key Features

- **Machine Learning Models**: Naive Bayes, SVM, Logistic Regression, Random Forest, and XGBoost.
- **Feature Engineering**: Includes TF-IDF vectorization, character-level n-grams, and contextual embeddings.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix analysis.

## Prerequisites

To run this project, you need:
- Python 3.8 or later
- Jupyter Notebook
- Required Python libraries (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Bangla-Homograph-Disambiguation.git
   cd Bangla-Homograph-Disambiguation
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Run the notebook file: **`Homograph.ipynb`**

## Dataset

The dataset consists of sentences containing the homographs mentioned above. These were derived from sources such as the SUMono corpus, web scraping of blogs, news articles, and social media.

- **Features**: Contextual n-grams, TF-IDF scores, and word embeddings.
- **Target**: Correct classification of homographs.

## Project Structure

```
Bangla-Homograph-Disambiguation/
├── Homograph.ipynb                  # Main notebook for analysis
├── Homograph_Paper_3rd_Draft.docx   # Detailed research paper
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
```

## Evaluation Metrics

The models were evaluated using:
- **Accuracy**: Overall classification accuracy.
- **Precision and Recall**: Measure of relevance and completeness.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: Breakdown of correct and incorrect classifications.

## Results

SVM emerged as the top-performing model across all datasets, demonstrating high accuracy and robustness, especially after feature engineering and hyperparameter tuning.

- **Best Model**: SVM
- **Feature Engineering**: Significant improvement in performance metrics across all models.
- **Key Insights**: Contextual embeddings and n-grams are vital for handling Bangla’s rich morphology.

## Key Technologies

- **Python**: Programming language.
- **Scikit-learn**: Machine learning model implementation.
- **Pandas**: Data manipulation.
- **Seaborn/Matplotlib**: Visualization.
- **TF-IDF and Word2Vec**: Feature extraction techniques.

## Future Work

- Explore advanced NLP models like BERT and LSTM for better context understanding.
- Address dataset class imbalance using techniques like SMOTE.
- Integrate ensemble learning methods for improved performance.

## License

This project is licensed under the [MIT License](LICENSE).

## Authors

- **Sayma Sultana Chowdhury**: Shahjalal University of Science and Technology
- **Md Kamruzzaman Kamrul**: Georgia State University
- **Eashraque Jahan Easha**: University of Denver
