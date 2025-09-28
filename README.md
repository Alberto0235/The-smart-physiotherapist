# 🤖 The Smart Physiotherapist – Human Activity Recognition (HAR) via UWB Radar

**Course/Context:** Hardware Architectures For Embedded and Edge AI Lab Project (Politecnico di Milano)
**Authors:** Alberto Marrone and Etienne Marie Rault
**Academic Year:** 2024-25

---

## 📌 Project Overview

This repository hosts all the files related to the **"The Smart Physiotherapist"** project.

The project focuses on developing an innovative, non-invasive system for **Human Activity Recognition (HAR)** of specific motor activities, primarily aimed at **remote patient monitoring** and assistance in rehabilitation contexts. The system leverages **Ultra-Wideband (UWB) radar sensors** to acquire motion data, which is then processed through an advanced **TinyML (Machine Learning on Edge Devices)** pipeline. The core objective is to achieve high classification accuracy while adhering to the strict memory and processing time constraints of resource-limited Microcontroller Units (MCUs).

---

## 💡 Methodology and Technical Details

The project methodology is divided into three key phases: acquisition, pre-processing, and classification.

### 1. Recognized Activities
The system was trained to recognize five fundamental movement classes crucial for physiotherapy and rehabilitation:
* **"left arm up"**
* **"right arm up"**
* **"left leg up"**
* **"right leg up"**
* **"still position"** (baseline)

### 2. Data Acquisition and Pre-processing
* **Raw Data:** Acquired using two distinct 3-antenna UWB radar platforms: the **SR250Mate** and an **Infineon sensor**. Raw data is stored as complex-valued spatio-temporal matrices in **NumPy** (`.npy`) files.
* **Custom Python Pipeline:** A dedicated **Python** script transforms the raw radar data into a visual format (grayscale images) optimized for Convolutional Neural Networks (CNNs). This process includes:
    * Calculating the signal magnitude.
    * Selecting the 40 most relevant "range bins" (corresponding to the subject at 1 meter).
    * Horizontally concatenating specific antenna data streams (e.g., RX0+RX1+RX2 for 120 pixels wide images) to create the final input image.

### 3. Model Development (TinyML)
Neural Network models were trained and optimized for efficient inference on edge devices. The work includes a comparative analysis, confirming the superior performance achieved with the Infineon sensor.

| Dataset Used | Final Accuracy (Test) | Notes |
| :--- | :--- | :--- |
| Infineon (RX0+RX1+RX2) | **97.03%** | **Best Performing Model.** Confirms the superior data quality of the Infineon sensor and the added value of the third antenna stream. |
| Infineon (RX0+RX1) | 94.12% | Excellent accuracy, but slightly lower than the full 3-antenna dataset. |
| SR250Mate (All Sets) | 10.89% - 16.83% | Significantly lower performance, attributed to the sensor's intrinsically lower Signal-to-Noise Ratio (SNR). |

---

## 🛠️ Tools and Technologies Used

This project combines state-of-the-art hardware with specialized software platforms for Edge AI development:

| Category | Tool / Technology | Specific Usage |
| :--- | :--- | :--- |
| **Acquisition Hardware** | **Infineon** 3-Antenna UWB Radar | Primary platform for high-quality data collection. |
| | **SR250Mate** 3-Antenna UWB Radar | Used for comparative data analysis. |
| **ML/AI Platform** | **Edge Impulse** | Training, optimization, and profiling of models for TinyML deployment. |
| **Languages & Libraries** | **Python** | Development of the data pre-processing pipeline. |
| | **NumPy** | Raw data format and multi-dimensional array manipulation. |
| **Target Deployment** | **Arduino Nano 33 BLE Sense** | Considered for initial testing and Edge AI compatibility. |


---

## 📂 Repository Structure

The repository is organized to provide comprehensive access to all project components:

* `\Infineon\` – Contains the raw and pre-processed dataset collected with the Infineon sensor, along with specific Python functions for its elaboration.
* `\SR250mate\` – Contains the raw and pre-processed dataset collected with the SR250mate sensor, along with specific Python functions for its elaboration.
* `Function_prepare_Final_compact_form.py` – General Python function used as a base for the initial data pre-processing stages.
* `Function_prepare_Infineon.py` – Specific Python function for preprocessing the Infineon data.
* `The_smart_physiotherapist.pdf` – The full project report detailing the methodology, comparative analysis, and results.
* `merged_dataset.zip` – An extended, pre-processed dataset used for training, integrating additional files.

---

## 🔗 Edge Impulse Models (Impulses)

Direct links to the trained and optimized models (Impulses) on the Edge Impulse platform are provided below:

* **Impulse 1 (Accurate):** The final, high-accuracy model (93.8% training, 97.03% test on Infineon RX0+RX1+RX2).
    * [Link Edge Impulse Accurate Model](https://studio.edgeimpulse.com/studio/740207/impulse/4/learning/keras/12)
* **Impulse 2 (Optimized/Merged):** An optimized version, likely based on a merged dataset or a lighter architecture.
    * [Link Edge Impulse Optimized Model](https://studio.edgeimpulse.com/studio/741999/impulse/1/learning/keras/4)

---

## ✅ Key Results and Conclusions

* **Technology Validation:** The project successfully demonstrated that UWB radar technology is highly effective for HAR in a rehabilitation context, offering high precision and privacy.
* **Sensor Quality:** A clear correlation was established between the quality of the raw sensor data (Infineon vs. SR250Mate) and the final model accuracy, a critical insight for Edge AI projects.
* **Functional TinyML:** Despite the complexity of the dataset, the optimized model achieved exceptional accuracy (>97%) while remaining compatible for deployment on high-end Microcontrollers (STM32).
