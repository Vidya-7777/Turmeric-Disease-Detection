# Turmeric Disease Detection: Deep Learning Meets Plant Health

This project showcases the application of deep learning and computer vision in plant disease detection, with a beautiful web interface for end-users.  
It is an intelligent system built to detect **turmeric plant diseases** from leaf images, segment the infected area, and suggest expert remedies — making plant care smarter and more accessible.

---

## About the Project

This project presents a real-world application of deep learning, built with PyTorch and deployed through a Flask web server. It combines plant pathology and artificial intelligence to provide turmeric farmers or researchers a practical solution for monitoring plant health.

We developed an interactive web-based disease diagnosis platform that:

- Classifies turmeric leaf images into one of three disease types
- Segments the infected area for visual feedback
- Suggests customized remedies (cultural, biological, and chemical)
- Provides a simple, user-friendly experience through a responsive web interface

It demonstrates how **AI can revolutionize agriculture** with accessible and automated plant care systems.

---

## Dataset Access

The turmeric plant leaf disease dataset used in this project is publicly available and free to use. It contains images categorized into classes like **Rhizome Rot**, **Leaf Blotch**, **Dry Leaf**, and **Healthy Leaf**, including both original and augmented images.

**Download Dataset Here:** 
[**Turmeric Plant Disease Dataset – Mendeley Data**](https://data.mendeley.com/datasets/g46dvrcvwn/1?utm_)  

---

## How It Works

**Image Upload = Plant Leaf Sample**  
Users upload an image of the turmeric leaf using the intuitive drag-and-drop interface.

**Model = Disease Expert**  
The uploaded image is passed through a trained deep learning model built using a lightweight transformer-inspired encoder and CNN-based decoder.

**Prediction + Segmentation**  
The app shows:
- The predicted disease name
- A segmented image of the infected region
- A comprehensive remedy plan for the disease

**Remedy Suggestions = Expert Advice**  
Each detected disease comes with multi-layered remedy guidance (cultural, chemical, biological).

---

## Disease Categories

The model supports the following classes:

- **Rhizome Rot Disease**
- **Leaf Blotch**
- **Dry Leaf Syndrome**

Each class includes customized remedies for practical action.

---

## Tech Stack

| Component       | Technology Used         |
|----------------|--------------------------|
| **Frontend**    | HTML, CSS (custom + Bootstrap), JS |
| **Backend**     | Python Flask             |
| **Modeling**    | PyTorch (CNN + Transformer encoder) |
| **CV & Processing** | OpenCV, NumPy           |
| **Deployment**  | Local Flask App          |

---

## Unique Features

- **Efficient model**: Lightweight transformer-based encoder with high accuracy
- **Image Segmentation**: Visual feedback of affected leaf region
- **Custom Remedies**: Integrated multi-type remedies for each disease
- **Practical Use Case**: Created for turmeric farmers, agritech startups, and students
- **Attractive UI**: Modern, responsive interface with drag-and-drop upload

---
