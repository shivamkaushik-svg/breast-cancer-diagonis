# 🔬 Breast Cancer Detection System (Deep Learning)

## 📖 About the Project
The **Breast Cancer Detection System** is a deep learning web application designed to classify cell characteristics as either **Benign** or **Malignant**. It utilizes a trained **TensorFlow/Keras Neural Network** to analyze a 30-feature medical profile. 

To enhance user experience, the web dashboard requires only the 10 core "Mean" metrics to be inputted via interactive sliders, while the application's backend automatically calculates and appends the remaining 20 standard "Error" and "Worst" metrics to run the full neural network analysis.

⚠️ **Important Note:** This repository is specifically structured and configured for **Render Deployment/Hosting**. 

If you are looking for the complete deep learning workflow—including data exploration, scaling, neural network architecture, and model training—please visit the Google Colab Notebook linked below:
👉 **[Full Source Code & Dataset (Google Colab)](https://colab.research.google.com/drive/1o5cNDeLDWfuDZynvWLdEB6sa_40dLPf6?usp=sharing)**

## 👨‍💻 Developer
**Chandan Saroj** | MERN Stack Developer & SDE
* **LinkedIn:** [Connect with me](https://www.linkedin.com/in/chandan-saroj/)
* **GitHub:** [Check out my projects](https://github.com/chandanXP)

## 🛠️ Tools & Technologies Used
* **Deep Learning:** TensorFlow / Keras (Sequential Neural Network)
* **Data Processing:** Scikit-learn (StandardScaler), NumPy, Pandas
* **Web UI Framework:** Gradio
* **Cloud Hosting:** Render
* **Development Environment:** Google Colab / Jupyter

---

## 🚀 How to Host Your Own Model on Render
You can use this repository as a template to host your own deep learning models on the web for free. Follow these step-by-step instructions:

### Step 1: Prepare Your Code
1. **Clone this repository** to your local machine.
2. **Push the code** to a new, public repository on your personal GitHub account.
3. **Train your model** (you can use the provided Colab link above). You must save **both** your scaler and your neural network.
4. Place **both** your `breast_cancer_model.h5` (the neural network) and `breast_cancer_model.pkl` (the scaler) files into the root of your cloned repository.
5. **Important for Deep Learning:** Ensure your `requirements.txt` file explicitly includes `tensorflow` and `scikit-learn==1.6.1`, otherwise the server will fail to boot!

### Step 2: Deploy on Render
1. Go to [Render.com](https://render.com/) and Log In or Sign Up.
2. Click on the **New** button in the dashboard and select **Web Service**.
3. Connect your GitHub account and select the repository you just created.
4. Fill in the deployment details:
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `python app.py`
5. Scroll down and select the **Free** instance type.
6. Click **Create Web Service**. 

Boom! 💥 Render will build your environment and give you a live URL. Your deep learning project is now live on the internet!

---

## 🌐 Live Demo
Check out the live, working version of this project hosted on Render:
👉 **[Breast Cancer Detection System - Live App](https://breast-cancer-diagnose-system.onrender.com/)**
