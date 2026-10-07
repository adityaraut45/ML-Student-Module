import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(
    page_title="Student ML Module",
    page_icon="🎓",
    layout="wide"
)

DATA_FILE = "dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

st.title("🎓 Student Performance ML Module")
st.write("Supervised and Unsupervised Machine Learning using the student dataset.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose Module",
    ["🏠 Dataset Overview", "🤖 Supervised Learning",
     "🔵 Unsupervised Learning", "📊 Visualizations"]
)

# ---------------- DATA OVERVIEW ----------------
if page == "🏠 Dataset Overview":
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Pass", int((df["result"] == "Pass").sum()))
    c4.metric("Fail", int((df["result"] == "Fail").sum()))

    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing, use_container_width=True)

    st.info(
        "The app automatically handles missing numeric values with the median "
        "and missing categorical values with the most frequent category."
    )

# ---------------- HELPERS ----------------
def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

def get_features_for_target(target):
    # student_id is only an identifier and should not train a model.
    # final_score is excluded when predicting result to avoid target leakage.
    excluded = ["student_id", target]

    if target == "result":
        excluded.append("final_score")
    elif target == "final_score":
        excluded.append("result")

    return [c for c in df.columns if c not in excluded]

# ---------------- SUPERVISED ----------------
if page == "🤖 Supervised Learning":
    st.header("🤖 Supervised Learning")
    st.write(
        "The model learns from labelled student data and predicts either "
        "the student's result or final score."
    )

    target = st.selectbox(
        "What do you want to predict?",
        ["result", "final_score"]
    )

    features = get_features_for_target(target)

    st.write("### Features used by the model")
    st.write(", ".join(features))

    if target == "result":
        algorithm = st.selectbox(
            "Choose Classification Algorithm",
            ["Logistic Regression", "Decision Tree", "KNN"]
        )
    else:
        algorithm = st.selectbox(
            "Choose Regression Algorithm",
            ["Linear Regression", "Decision Tree"]
        )

    test_size = st.slider("Test data (%)", 10, 40, 20)

    X = df[features]
    y = df[target]

    if st.button("🚀 Train Model", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size / 100,
            random_state=42,
            stratify=y if target == "result" else None
        )

        preprocessor = make_preprocessor(X_train)

        if target == "result":
            if algorithm == "Logistic Regression":
                model = LogisticRegression(max_iter=2000)
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42, max_depth=5)
            else:
                model = KNeighborsClassifier(n_neighbors=5)

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            accuracy = accuracy_score(y_test, pred)
            st.success("Model trained successfully!")

            a, b = st.columns(2)
            a.metric("Accuracy", f"{accuracy * 100:.2f}%")
            b.metric("Test Records", len(y_test))

            st.subheader("Classification Report")
            report = classification_report(
                y_test, pred, output_dict=True, zero_division=0
            )
            st.dataframe(
                pd.DataFrame(report).transpose(),
                use_container_width=True
            )

        else:
            if algorithm == "Linear Regression":
                model = LinearRegression()
            else:
                model = DecisionTreeRegressor(
                    random_state=42, max_depth=5
                )

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, pred)

            st.success("Model trained successfully!")

            a, b, c, d = st.columns(4)
            a.metric("MAE", f"{mae:.2f}")
            b.metric("MSE", f"{mse:.2f}")
            c.metric("RMSE", f"{rmse:.2f}")
            d.metric("R²", f"{r2:.3f}")

        # Store trained model in session
        st.session_state["trained_pipe"] = pipe
        st.session_state["target"] = target
        st.session_state["features"] = features
        st.session_state["algorithm"] = algorithm

    # ---------------- PREDICTION ----------------
    if "trained_pipe" in st.session_state:
        st.divider()
        st.subheader("🔮 Predict for a New Student")

        if (
            st.session_state["target"] == target
            and st.session_state["features"] == features
        ):
            input_data = {}

            col1, col2 = st.columns(2)

            for i, feature in enumerate(features):
                container = col1 if i % 2 == 0 else col2

                if feature == "age":
                    input_data[feature] = container.number_input(
                        "Age", min_value=10, max_value=100, value=20
                    )

                elif feature == "study_hours_per_day":
                    input_data[feature] = container.number_input(
                        "Study Hours Per Day", 0.0, 24.0, 5.0, 0.1
                    )

                elif feature == "attendance_percent":
                    input_data[feature] = container.number_input(
                        "Attendance (%)", 0.0, 100.0, 75.0, 0.1
                    )

                elif feature == "previous_score":
                    input_data[feature] = container.number_input(
                        "Previous Score", 0.0, 100.0, 60.0, 0.1
                    )

                elif feature == "sleep_hours":
                    input_data[feature] = container.number_input(
                        "Sleep Hours", 0.0, 24.0, 7.0, 0.1
                    )

                elif feature == "gender":
                    input_data[feature] = container.selectbox(
                        "Gender", sorted(df["gender"].dropna().unique())
                    )

                elif feature == "internet_access":
                    input_data[feature] = container.selectbox(
                        "Internet Access", sorted(df["internet_access"].dropna().unique())
                    )

                elif feature == "parental_education":
                    input_data[feature] = container.selectbox(
                        "Parental Education",
                        sorted(df["parental_education"].dropna().unique())
                    )

                elif feature == "extracurricular":
                    input_data[feature] = container.selectbox(
                        "Extracurricular",
                        sorted(df["extracurricular"].dropna().unique())
                    )

            if st.button("🎯 Predict", type="primary"):
                new_student = pd.DataFrame([input_data])
                prediction = st.session_state["trained_pipe"].predict(new_student)[0]

                if target == "result":
                    st.success(f"Predicted Result: **{prediction}**")

                    if hasattr(st.session_state["trained_pipe"], "predict_proba"):
                        try:
                            probability = (
                                st.session_state["trained_pipe"]
                                .predict_proba(new_student)[0].max() * 100
                            )
                            st.info(
                                f"Prediction confidence: {probability:.2f}%"
                            )
                        except Exception:
                            pass
                else:
                    st.success(
                        f"Predicted Final Score: **{float(prediction):.2f} / 100**"
                    )

# ---------------- UNSUPERVISED ----------------
elif page == "🔵 Unsupervised Learning":
    st.header("🔵 Unsupervised Learning - K-Means")

    st.write(
        "K-Means groups students with similar characteristics without using "
        "the Result column."
    )

    cluster_features = st.multiselect(
        "Select features for clustering",
        [
            "age",
            "study_hours_per_day",
            "attendance_percent",
            "previous_score",
            "sleep_hours"
        ],
        default=[
            "study_hours_per_day",
            "attendance_percent",
            "previous_score"
        ]
    )

    k = st.slider("Number of clusters", 2, 6, 3)

    if len(cluster_features) < 2:
        st.warning("Please select at least 2 features.")
    elif st.button("🔵 Run K-Means", type="primary"):
        data = df[cluster_features].copy()
        data = data.fillna(data.median(numeric_only=True))

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(scaled)

        result = df[["student_id", "result"]].copy()
        result["Cluster"] = labels

        st.success("Clustering completed!")

        st.subheader("Cluster Results")
        st.dataframe(result, use_container_width=True)

        st.subheader("Cluster Summary")
        summary = data.copy()
        summary["Cluster"] = labels
        st.dataframe(
            summary.groupby("Cluster")[cluster_features].mean().round(2),
            use_container_width=True
        )

        if len(cluster_features) >= 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(
                data[cluster_features[0]],
                data[cluster_features[1]],
                c=labels
            )
            ax.set_xlabel(cluster_features[0])
            ax.set_ylabel(cluster_features[1])
            ax.set_title("K-Means Student Clusters")
            st.pyplot(fig)

# ---------------- VISUALIZATIONS ----------------
else:
    st.header("📊 Student Data Visualizations")

    chart = st.selectbox(
        "Choose visualization",
        [
            "Result Distribution",
            "Study Hours vs Final Score",
            "Attendance vs Final Score",
            "Previous Score vs Final Score"
        ]
    )

    if chart == "Result Distribution":
        counts = df["result"].value_counts()
        st.bar_chart(counts)

    elif chart == "Study Hours vs Final Score":
        st.scatter_chart(
            df[["study_hours_per_day", "final_score"]]
            .dropna()
            .set_index("study_hours_per_day")
        )

    elif chart == "Attendance vs Final Score":
        st.scatter_chart(
            df[["attendance_percent", "final_score"]]
            .dropna()
            .set_index("attendance_percent")
        )

    else:
        st.scatter_chart(
            df[["previous_score", "final_score"]]
            .dropna()
            .set_index("previous_score")
        )

st.divider()
st.caption("Student Performance ML Module | Built with Python, Streamlit and Scikit-learn")
