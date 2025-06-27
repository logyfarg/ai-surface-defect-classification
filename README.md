# 🟢 Surface Defect Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify industrial surface defects in steel images.

✅ **Dataset:**
[NEU Surface Defect Database](https://github.com/opensets/NEU-CLS)  
Classes include:
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

✅ **How to Run:**
1. Download the NEU dataset and place it in your project folder.
2. Install dependencies:
3. Run the training script or notebook:
4. The trained model will be saved as `surface_defect_model.h5`.

✅ **Model Summary:**
- Input size: 150x150 RGB images
- 2 Convolutional + MaxPooling layers
- Dense layers with dropout
- Softmax output over 6 defect classes

✅ **Result:**
Achieves good classification accuracy on validation data.

---

**Example prediction:**
![Example](https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Defect_example.png/320px-Defect_example.png)

---

Feel free to use or improve this project!
