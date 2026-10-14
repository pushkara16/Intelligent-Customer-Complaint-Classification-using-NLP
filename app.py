from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model
with open("customer_complaint_model.pkl", "rb") as file:
    model = pickle.load(file)

CONFIDENCE_THRESHOLD = 0.60  # 60%


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    status = None
    complaint = ""

    if request.method == "POST":
        complaint = request.form.get("complaint", "").strip()

        if complaint:
            # Predict the complaint category
            result = model.predict([complaint])
            prediction = str(result[0])

            # Check prediction confidence, if supported
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([complaint])[0]
                confidence = float(max(probabilities))

                if confidence < CONFIDENCE_THRESHOLD:
                    status = "Needs Further Checking"
                else:
                    status = "Prediction Available"
            else:
                # The model does not provide confidence probabilities
                status = "Needs Further Checking"
                confidence = None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        status=status,
        complaint=complaint
    )


if __name__ == "__main__":
    app.run(debug=True)
