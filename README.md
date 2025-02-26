# **Tracking U.S Land Cover Changes: A Dataset of Sentinel-2 Imagery and Dynamic World Labels**

## **Overview**
This repository provides a comprehensive workflow for **analyzing land cover changes** across the United States from **2016 to 2024** using **Sentinel-2 imagery** and **Dynamic World annotations**. It includes tools for **downloading satellite images, preprocessing data, training deep learning models for segmentation, and evaluating land cover changes**.

This repository is designed to be **accessible to users without prior knowledge of artificial intelligence**, offering easy-to-use scripts and notebooks to facilitate land cover analysis.

## **Datasets Used**
### **1. Dynamic World (DW)**
Dynamic World is a high-resolution land cover dataset developed by **Google** and the **World Resources Institute**. It provides **near real-time land classification maps** based on **Sentinel-2 imagery**, with the following **nine land cover categories**:
- Water
- Trees
- Grass
- Crops
- Shrubland
- Built-up areas
- Bare ground
- Snow/Ice
- Flooded vegetation

This dataset is essential for monitoring **urbanization, deforestation, agriculture, and climate change**.

### **2. Sentinel-2**
Sentinel-2 is a **multispectral satellite mission** from the **European Space Agency (ESA)**, widely used for:
- **Land cover classification**
- **Forestry and agriculture monitoring**
- **Urban expansion analysis**
- **Disaster monitoring**

This repository leverages Sentinel-2 imagery **(bands: Red, Green, Blue, NIR, SWIR1, SWIR2)** to enable **detailed environmental monitoring**.

## **Project Workflow**
This repository contains **Jupyter Notebooks** and **Python scripts** to guide users through every step of the process.

### **1. Downloading Satellite Images**
- A **Jupyter Notebook** allows users to download **Sentinel-2 and Dynamic World composites** from **Google Earth Engine (GEE)**.
- The images are saved in **.tif format** for further processing.

### **2. Data Preprocessing**
- Converts the downloaded `.tif` images into **HDF5 (.H5) format** for deep learning models.
- Normalizes the data using **Z-score normalization** to improve model performance.
- Applies **Mosaic Augmentation** to enhance training data diversity.

### **3. Land Cover Analysis Notebook**
- Computes **pixel counts per land cover category** from **Dynamic World** data.
- Generates **bar charts** to visualize land cover distribution.
- Allows users to **highlight specific categories** (e.g., urban areas).
- Analyzes **Sentinel-2 band distributions** and generates **histograms**.
- Clips and merges raster datasets to study land cover changes over time.

### **4. Training and Evaluating Deep Learning Models**
- **Segmentation models** trained to predict land cover categories using Sentinel-2 RGB bands:
  - **FCN (Fully Convolutional Network)**
  - **LR-ASPP (Lite Reduced Atrous Spatial Pyramid Pooling)**
- Evaluates model performance on test data.
- Produces segmentation maps for visual analysis.

## **Installation & Usage**
### **1. Prerequisites**
Install the required dependencies using:
```bash
pip install geopandas rasterio matplotlib numpy tqdm h5py tensorflow
```
### **2. Clone the Repository**
```bash
git clone 
```
### **3. Download Satellite Data**
Open Jupyter Notebook and run
``` bash
Download_data_from_GEE.ipynb
```

### **4. Process and Analyze the Data**
Open Jupyter Notebook and run
``` bash
Analyze land cover changes.ipynb
```
### **5. Train and Evaluate Segmentation Models**
Run the provided scripts to train the deep learning models
``` bash
python Train_model.py
python eval_model.py
```

---

### **Resultados y Créditos**

## **Results & Outputs**
This repository generates:
- **Clipped raster images** of selected study areas.
- **Land cover change statistics** (pixel counts per category).
- **Histograms of Sentinel-2 band distributions**.
- **Segmentation maps predicted by deep learning models**.
- **Trained FCN and LR-ASPP models for land cover classification**.

## **Contributors**
- **Antonio Rangel**
- **Juan Terven**

## **Data Sources**
- **Dynamic World (Google & World Resources Institute)**
- **Sentinel-2 (European Space Agency - ESA)**
- **U.S. Census Bureau - [Historical Apportionment Data Map](https://www.census.gov/library/visualizations/interactive/historical-apportionment-data-map.html)**


## **Acknowledgments**
This repository is part of a research project to facilitate land cover change analysis and AI-based segmentation for environmental monitoring.

