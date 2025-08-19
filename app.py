from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Loading models 
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load dataset
DATA_PATH = "access_logs.csv"
df = pd.read_csv(DATA_PATH)

df["time_of_access"] = pd.to_numeric(df["time_of_access"], errors="coerce")
df["time_of_day"] = df["time_of_access"].fillna(0).astype(int)

# Mapping 0 and 1 as allowed or denied to increase understanding 
outcome_map = {1: "Allowed", 0: "Denied"}
df["outcome"] = df["access_granted"].map(outcome_map)

app = Flask(__name__)

# Risk Calculation 
def calculate_risks(context):
    user_id = context.get("user_id", "")
    device = context.get("device_type", "").lower()
    location = context.get("location", "").lower()
    resource = context.get("resource_requested", "").lower()

    user_risk = 0.2 if user_id.startswith("U0") else 0.4
    device_risk = 0.3 if device in ["laptop", "desktop"] else 0.6
    location_risk = 0.2 if location in ["india", "us"] else 0.5
    resource_risk = 0.3 if resource in ["server_a", "db1"] else 0.6

    return {
        "user_risk": round(user_risk, 2),
        "device_risk": round(device_risk, 2),
        "location_risk": round(location_risk, 2),
        "resource_risk": round(resource_risk, 2),
        "user_device_risk": round((user_risk + device_risk) / 2, 2),
        "user_location_risk": round((user_risk + location_risk) / 2, 2),
        "device_location_risk": round((device_risk + location_risk) / 2, 2),
    }

# Prediction 
def predict_access(context):
    risks = calculate_risks(context)

    df_context = pd.DataFrame([context])
    df_context["time_of_access"] = pd.to_numeric(df_context["time_of_access"], errors="coerce")
    df_context["time_of_day"] = df_context["time_of_access"].fillna(0).astype(int)

    X_struct = preprocessor.transform(df_context)
    X_inter = np.array([[risks['user_device_risk'],
                         risks['user_location_risk'],
                         risks['device_location_risk']]])
    X_text = tfidf.transform([context['intent_prompt']]).toarray()
    X = np.hstack([X_struct, X_inter, X_text])

    proba = model.predict_proba(X)[0, 1]
    decision = int(proba >= 0.5)
    trust_score = proba * 100

    return decision, trust_score, risks

# the routes to access all the files 
@app.route("/")
def home():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    # Access patterns by hour
    fig_time = px.histogram(
        df,
        x="time_of_day",
        color="outcome",   
        category_orders={"time_of_day": list(range(24))},
        title="Access Patterns by Hour",
        barmode="group"
    )
    fig_time.update_xaxes(
        title="Hour of Day",
        tickmode="array",
        tickvals=list(range(24)),
        ticktext=[f"{h:02d}:00" for h in range(24)],
        range=[0, 23]
    )
    fig_time.update_yaxes(title="Count")
    time_chart = pio.to_html(fig_time, full_html=False)

    # Access by device 
    fig_device = px.histogram(
        df,
        x="device_type",
        color="outcome",  
        title="Access by Device",
        barmode="group"
    )
    fig_device.update_xaxes(title="Device Type")
    fig_device.update_yaxes(title="Count")
    device_chart = pio.to_html(fig_device, full_html=False)

    # Resource access frequency 
    fig_resource = px.histogram(
        df,
        x="resource_requested",
        color="outcome",   
        title="Resource Access Frequency",
        barmode="group"
    )
    fig_resource.update_xaxes(title="Resource")
    fig_resource.update_yaxes(title="Count")
    resource_chart = pio.to_html(fig_resource, full_html=False)

    # Device distribution 
    fig_device_pie = px.pie(
        df,
        names="device_type",
        title="Device Distribution",
        hole=0.0
    )
    device_pie_chart = pio.to_html(fig_device_pie, full_html=False)

    # Access outcome 
    fig_outcome = px.pie(
        df,
        names="outcome",
        title="Access Outcomes",
        hole=0.5,
        color="outcome",
        color_discrete_map={"Allowed": "green", "Denied": "red"}
    )
    outcome_chart = pio.to_html(fig_outcome, full_html=False)

    return render_template(
        "dashboard.html",
        time_chart=time_chart,
        device_chart=device_chart,
        resource_chart=resource_chart,
        device_pie_chart=device_pie_chart,
        outcome_chart=outcome_chart
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        context = {
            "user_id": request.form["user_id"],
            "device_type": request.form["device_type"],
            "location": request.form["location"],
            "time_of_access": request.form["time_of_access"],
            "resource_requested": request.form["resource_requested"],
            "intent_prompt": request.form["intent_prompt"]
        }

        decision, trust_score, risks = predict_access(context)

        result = {
            "decision": "ACCESS GRANTED" if decision == 1 else "ACCESS DENIED",
            "trust_score": round(trust_score, 2),
            "risks": risks,
            "request": context
        }

        return render_template("result.html", result=result)

    return render_template("index.html")

@app.route("/dataset")
def dataset():
    return render_template("data.html")

@app.route("/get_data")
def get_data():
    df_copy = df.copy()
    df_copy["hour"] = df_copy["time_of_day"].apply(
        lambda h: f"{int(h):02d}:00" if pd.notnull(h) else "NA"
    )
    data = df_copy.to_dict(orient="records")
    return jsonify({"data": data})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
