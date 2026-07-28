import os
import joblib
import traceback
import numpy as np
import gradio as gr
import tensorflow as tf

# ==========================================================
# Load Model & Scaler
# ==========================================================

try:
    scaler = joblib.load("breast_cancer_scaler.pkl")
    deployed_nn = tf.keras.models.load_model("breast_cancer_model.h5")
    print("✅ Scaler and Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    scaler = None
    deployed_nn = None


# ==========================================================
# Prediction Function
# ==========================================================

def predict_cancer(*features):

    values = list(features)

    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill all 30 input values."

    try:
        values = [float(v) for v in values]

    except Exception:
        return "❌ Invalid input. Only numeric values are allowed."

    if scaler is None or deployed_nn is None:
        return "❌ Model failed to load."

    try:

        input_array = np.array([values])

        scaled = scaler.transform(input_array)

        prediction = deployed_nn.predict(scaled, verbose=0)[0][0]

        if prediction >= 0.5:

            return (
                f"🟢 Prediction Confidence : {prediction:.2%}\n\n"
                "Classification : BENIGN\n\n"
                "The tumor is predicted to be NON-CANCEROUS."
            )

        else:

            return (
                f"🔴 Prediction Confidence : {(1-prediction):.2%}\n\n"
                "Classification : MALIGNANT\n\n"
                "The tumor is predicted to be CANCEROUS.\n\n"
                "Please consult an oncologist."
            )

    except Exception:

        return traceback.format_exc()


# ==========================================================
# Interface
# ==========================================================

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
# 🔬 Breast Cancer Detection System

### Deep Learning Based Prediction
"""
    )

    with gr.Tabs():

        with gr.Tab("Mean Features"):

            with gr.Row():

                with gr.Column():
                    f1 = gr.Number(label="Mean Radius")
                    f2 = gr.Number(label="Mean Texture")
                    f3 = gr.Number(label="Mean Perimeter")
                    f4 = gr.Number(label="Mean Area")
                    f5 = gr.Number(label="Mean Smoothness")

                with gr.Column():
                    f6 = gr.Number(label="Mean Compactness")
                    f7 = gr.Number(label="Mean Concavity")
                    f8 = gr.Number(label="Mean Concave Points")
                    f9 = gr.Number(label="Mean Symmetry")
                    f10 = gr.Number(label="Mean Fractal Dimension")

        with gr.Tab("Error Features"):

            with gr.Row():

                with gr.Column():
                    f11 = gr.Number(label="Radius Error")
                    f12 = gr.Number(label="Texture Error")
                    f13 = gr.Number(label="Perimeter Error")
                    f14 = gr.Number(label="Area Error")
                    f15 = gr.Number(label="Smoothness Error")

                with gr.Column():
                    f16 = gr.Number(label="Compactness Error")
                    f17 = gr.Number(label="Concavity Error")
                    f18 = gr.Number(label="Concave Points Error")
                    f19 = gr.Number(label="Symmetry Error")
                    f20 = gr.Number(label="Fractal Dimension Error")

        with gr.Tab("Worst Features"):

            with gr.Row():

                with gr.Column():
                    f21 = gr.Number(label="Worst Radius")
                    f22 = gr.Number(label="Worst Texture")
                    f23 = gr.Number(label="Worst Perimeter")
                    f24 = gr.Number(label="Worst Area")
                    f25 = gr.Number(label="Worst Smoothness")

                with gr.Column():
                    f26 = gr.Number(label="Worst Compactness")
                    f27 = gr.Number(label="Worst Concavity")
                    f28 = gr.Number(label="Worst Concave Points")
                    f29 = gr.Number(label="Worst Symmetry")
                    f30 = gr.Number(label="Worst Fractal Dimension")

    gr.Markdown("---")

    with gr.Row():

        submit_btn = gr.Button(
            "Run Prediction",
            variant="primary"
        )

        clear_btn = gr.ClearButton()

    result_box = gr.Textbox(
        label="Prediction Result",
        lines=8
    )

    gr.Markdown(
        """
---

## 👨‍💻 Developer

**Created By:** Shivam Kaushik

**GitHub:** https://github.com/shivamkaushik-svg

**LinkedIn:** https://www.linkedin.com/in/shivam-kaushik-87000b3a8

**Instagram:** https://www.instagram.com/shivamkaushik_178

"""
    )

    inputs = [
        f1, f2, f3, f4, f5,
        f6, f7, f8, f9, f10,
        f11, f12, f13, f14, f15,
        f16, f17, f18, f19, f20,
        f21, f22, f23, f24, f25,
        f26, f27, f28, f29, f30,
    ]

    submit_btn.click(
        fn=predict_cancer,
        inputs=inputs,
        outputs=result_box,
    )

    clear_btn.add(
        inputs + [result_box]
    )


# ==========================================================
# Launch
# ==========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
