## Satellite Imagery–Based Property Valuation 

# Project Overview
Property valuation is traditionally driven by structured attributes such as size, location, and construction quality. In this project, we explore whether satellite imagery can provide additional neighborhood-level context and improve price prediction when combined with tabular housing data.
A multimodal regression pipeline was built to integrate structured features with CNN-based visual embeddings extracted from satellite images. Several modeling strategies were evaluated to understand when multimodal learning helps and when it does not.


# Dataset

- Primary Dataset:
King County House Sales Dataset
Source: Kaggle – House Sales in King County, USA
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
Target Variable: price
Key Tabular Features:
- Number of bedrooms and bathrooms
- Living area and lot size
- Construction grade and condition
- View and waterfront indicators
- Geographic coordinates (latitude & longitude)
These features capture both property-level characteristics and location-based value drivers.

- Satellite Imagery
Satellite images were programmatically fetched using geographic coordinates (latitude and longitude). The images provide overhead neighborhood context such as:
- Road density
- Surrounding buildings
- Green cover
- Proximity to water bodies
Each image was processed and converted into numerical embeddings using a pretrained Convolutional Neural Network (CNN).


# Methodology

1. Tabular Modeling
The following regression models were evaluated using structured features only:
- Linear Regression
- Random Forest Regressor (RF)
- Gradient Boosting Regressor (GBR)
Tree-based models (RF and GBR) were chosen due to their ability to model non-linear relationships common in real estate data.

2. Image Feature Extraction
- Satellite images resized to 224 × 224
- Feature extraction performed using a pretrained ResNet CNN
- Each image represented as a 512-dimensional embedding

3. Multimodal Fusion
To combine visual and tabular information:
Image embeddings were concatenated with structured features (late fusion)
Additional experiments included PCA-based dimensionality reduction and Ridge regression to control feature noise

4. Evaluation Metrics
Model performance was evaluated using:
- R² Score
- Root Mean Squared Error (RMSE)


# Results Summary: 

- Linear Regression underperformed due to its inability to model non-linear relationships.
- Random Forest and Gradient Boosting achieved strong performance using tabular data alone.
- Gradient Boosting Regressor (tabular-only) produced the best overall results, with the highest R² score.
- All multimodal models that incorporated satellite image embeddings showed degraded performance compared to tabular-only baselines.


# Analysis & Interpretation

The structured features already captured the dominant price-determining factors such as property size, construction quality, view, waterfront access, and geographic location. While satellite imagery provided interpretable neighborhood-level context, it did not contribute sufficient additional predictive signal.
The CNN-based image embeddings introduced high-dimensional noise, which outweighed any marginal visual benefit and led to reduced generalization performance.
Dimensionality reduction and regularization techniques were insufficient to recover useful predictive signal from satellite imagery.
This outcome highlights an important insight: multimodal learning does not always improve model performance, especially when the baseline structured data is already highly informative.


# Explainability (Grad-CAM — Conceptual)

Grad-CAM is a commonly used technique for interpreting convolutional neural networks by highlighting image regions that contribute most strongly to learned representations.
In this project, satellite images were processed using a pretrained CNN solely for feature extraction, and the regression models were trained independently on the extracted embeddings. As a result, Grad-CAM visualizations were not directly generated.
Conceptually, Grad-CAM applied to satellite imagery would be expected to emphasize coarse neighborhood-level features such as road networks, building density, green spaces, and nearby water bodies. These visual cues are intuitive indicators of neighborhood quality, but experimental results suggest that they do not provide sufficient additional predictive value beyond strong structured housing features.


# Final Conclusion

For this dataset, Gradient Boosting Regressor trained on structured tabular data provides the most accurate and reliable property price predictions. The inclusion of satellite image embeddings does not improve performance and, in several cases, degrades it. This project demonstrates the importance of modality relevance and empirical validation in multimodal machine learning systems.


# Repository Structure
.
├── data/
│   ├── raw/                 
│   ├── images/              
│   └── processed/           
├── data_fetcher.py          
├── property_valuation.ipynb 
├── 24114036_final.csv    
├── README.md
├── 24114036_report.pdf               


# How to Run the Project

1. Open property_valuation.ipynb
2. Run the notebook cells sequentially
3. Satellite images are downloaded using data_fetcher.py
4. Final predictions are saved as 24114036_final.csv


# Notes

- Multimodal learning was explored extensively but did not outperform strong tabular baselines.
- All results are reported transparently, highlighting both successes and limitations of the approach.
- While higher-resolution satellite imagery could potentially provide clearer visual details, the observed results suggest that overhead imagery,regardless of resolution,offers limited complementary predictive value when strong structured features are already available.