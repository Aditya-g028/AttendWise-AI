import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pathlib import Path

app = Flask(__name__)

# Load student dataset using an absolute project path
BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / "data" / "student_data.csv")

# Select input and output
X = data[["attendance", "internal_marks", "assignment_marks"]]
y = data["final_score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate model error
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)


@app.route("/")
def home():
    return render_template("index.html", error=round(error, 2))

@app.route("/predict", methods=["POST"])
def predict():

    attendance = float(request.form["attendance"])
    internal = float(request.form["internal_marks"])
    assignment = float(request.form["assignment_marks"])
    # Check if all values are between 0 and 100
    if not (0 <= attendance <= 100 and
        0 <= internal <= 100 and
        0 <= assignment <= 100):

        return render_template(
        "index.html",
        error=round(error, 2),
        error_message="Please enter values between 0 and 100."
    )

    student = pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal,
        "assignment_marks": assignment
    }])

    result = model.predict(student)
    score = round(result[0], 2)

    if score >= 80:
        message = "Excellent Performance!"
    elif score >= 60:
        message = "Good Performance!"
    else:
        message = "Needs Improvement!"

    return render_template(
    "index.html",
    prediction=score,
    message=message,
    error=round(error, 2)
    )


if __name__ == "__main__":
    app.run(debug=False)
