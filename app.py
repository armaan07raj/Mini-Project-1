# ============================================================
# CHUNK 1: imports and page setup
# ============================================================
import streamlit as st                                   # builds the web page
import pandas as pd                                      # data tables (DataFrames)
import numpy as np                                       # used once, for np.nan
import matplotlib.pyplot as plt                          # draws charts
import seaborn as sns                                    # nicer charts on top of matplotlib
from sklearn.model_selection import train_test_split     # splits data into train/test
from sklearn.linear_model import LinearRegression        # the OLS model
from sklearn.metrics import mean_squared_error           # measures model error

# Must be the FIRST st. command in the file
st.set_page_config(page_title="Telco Bill Explorer", layout="wide")


# ============================================================
# CHUNK 2: load and clean the data
# ============================================================
# @st.cache_data = "run this function once and remember the answer",
# so the cleaning doesn't re-run every time someone touches a widget.
@st.cache_data
def load_data():
    # Relative path: works on your laptop AND on Streamlit Cloud
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Missing values are hiding: 11 rows have " " (a blank string) in Total_Charges,
    # so isna() can't see them yet. Turn them into real NaN first (Week 3 slide).
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

    # Count them BEFORE dropping so we can justify the decision on the dashboard
    n_rows = len(df)
    n_missing = df["TotalCharges"].isna().sum()

    # Drop them: 11 rows is 0.16% of the data (rule of thumb: < 5% is fine to drop)
    df = df.dropna()

    # Total_Charges was text because of the blanks. Now that they're gone, make it a number.
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Derived column (required): tenure in years instead of months
    df["tenure_years"] = df["tenure"] / 12

    return df, n_rows, n_missing


df, n_rows, n_missing = load_data()


# ============================================================
# CHUNK 3: title (your central question)
# ============================================================
st.title("What drives a Telco customer's monthly bill, and do big spenders churn more?")
st.caption("Target we predict: Monthly Charges (dollars per month).")


# ============================================================
# CHUNK 4: sidebar widgets, filtering, and summary metrics
# ============================================================
# Everything inside "with st.sidebar:" shows up in the left panel
with st.sidebar:
    st.header("Filters")

    # Widget 1: multiselect. Returns a LIST of the chosen internet types.
    internet_options = list(df["InternetService"].unique())
    internet = st.multiselect("Internet service", internet_options, default=internet_options)

    # Widget 2: range slider. Returns a (low, high) pair.
    tmin, tmax = st.slider("Tenure (months)", 0, 72, (0, 72))

    # Widget 3: checkbox. Returns True or False.
    show_raw = st.checkbox("Show raw data")

# Boolean filtering (Week 2): one condition per widget, each in parentheses, joined with &
filtered = df[(df["InternetService"].isin(internet)) & (df["tenure"].between(tmin, tmax))]

# If the user filters everything away, say so and stop instead of crashing
if filtered.empty:
    st.warning("No customers match those filters.")
    st.stop()

# Summary metrics side by side (required: st.columns)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", len(filtered))
c2.metric("Avg monthly charge ($)", round(filtered["MonthlyCharges"].mean(), 2))
c3.metric("Avg tenure (months)", round(filtered["tenure"].mean(), 1))
c4.metric("Churn rate (% who left)", round((filtered["Churn"] == "Yes").mean() * 100, 1))

# Define the metric right where people see it
st.caption("**Churn rate** = the percentage of customers in the current filter who left the "
           "company (`Churn = Yes`). With no filters, 26.6% means about 1 in 4 customers left. "
           "It is recalculated whenever you move the sidebar filters.")

if show_raw:
    st.dataframe(filtered)


# ============================================================
# CHUNK 5: tabs + the Data & Cleaning tab
# ============================================================
tab1, tab2, tab3 = st.tabs(["Data & Cleaning", "Exploration", "Model"])

with tab1:
    st.header("Cleaning")
    st.write("Rows in raw file:", n_rows)
    st.write("Rows with missing Total Charges:", n_missing)
    st.write("Percent of rows affected:", round(n_missing / n_rows * 100, 2))
    st.markdown(
        "**Decision: drop.** Only 0.16% of rows are affected (well under 5%). All 11 are "
        "customers with `tenure = 0`, so they have not been billed yet. All 11 are `Churn = No`, "
        "but with only 11 rows that cannot bias the results."
    )

    # Required groupby with an aggregation
    st.subheader("Groupby: averages by internet service")
    by_internet = df.groupby("InternetService").agg(
        {"MonthlyCharges": "mean", "tenure": "mean", "customerID": "count"})
    st.dataframe(by_internet)
    st.caption("Fiber optic customers pay the most per month; customers with no internet pay the least.")

    # Required Faithfulness & Granularity section (three labeled parts)
    with st.expander("Faithfulness & Granularity"):
        st.markdown(
            """
**Granularity: what each row represents**
- Each row is **one customer** (one unique `customerID`): their services, contract, billing
  info, tenure in months, monthly charge, and whether they churned.
- The finest level is the customer, not individual bills, calls, or months. Each customer has
  a single `MonthlyCharges` number, so we cannot see how a bill changed over time.

**Scope: what the data does and does not cover**
- **Who:** customers of a single telecom company that sells home phone and internet service.
  The file has 7,043 customers and 21 columns (7,032 customers after cleaning).
- **When:** a single snapshot. There are no dates, so no time trends. The only time information
  is `tenure`, which ranges from 1 to 72 months (up to 6 years) after cleaning.
- **Where / who exactly:** there is no location column. The only demographics are gender,
  senior citizen, partner, and dependents.
- **Includes people who left:** about 26.5% of customers have `Churn = Yes`, so former customers
  are in the data, not only current ones.
- **What we can say:** how bills, services, and contracts are related for this company's customers.
- **What we cannot say:** that one thing *causes* another, or that the results hold for other
  companies, regions, or years. We also do not know how the company chose which customers to include.

**Strange values & faithfulness**
- `TotalCharges` was stored as text, with blank strings for brand-new customers (`tenure = 0`).
- `SeniorCitizen` is coded 0/1, while other yes/no columns use "Yes"/"No".
- The add-on columns (like `TechSupport`) use "No internet service" as a third value,
  which repeats `InternetService = No`.
- `tenure_years` is just `tenure / 12`, so the two columns carry the same information.
            """
        )
        st.write("Duplicate rows:", df.duplicated().sum())
        st.write("Duplicate customer IDs:", df["customerID"].duplicated().sum())
        st.write("Dataset shape after cleaning:", df.shape)


# ============================================================
# CHUNK 6: the Exploration tab (4 charts)
# ============================================================
# Every chart uses `filtered` (not `df`) so the sidebar widgets change them.
# Every chart gets its own fig, ax = plt.subplots() or charts stack on top of each other.
# Tabs inside the Exploration tab: one visualization per tab.
with tab2:
    v1, v2, v3, v4, v5 = st.tabs(
        ["Contract types", "Monthly charges", "Tenure vs. charges", "Correlations", "Churn and price"])

    with v1:
        chart, notes = st.columns([2, 1])   # chart on the left, explanation on the right
        with chart:
            # Bar plot: one categorical variable
            fig, ax = plt.subplots()
            sns.countplot(data=filtered, y="Contract", ax=ax)
            ax.set_title("Customers by contract type")
            ax.set_xlabel("Number of customers")
            st.pyplot(fig)
        with notes:
            st.subheader("What to notice")
            st.caption("Most customers are on month-to-month contracts.")

    with v2:
        chart, notes = st.columns([2, 1])
        with chart:
            # Histogram: one quantitative variable
            fig, ax = plt.subplots()
            sns.histplot(filtered["MonthlyCharges"], bins=30, ax=ax)
            ax.set_title("Distribution of monthly charges")
            ax.set_xlabel("Monthly charge ($)")
            st.pyplot(fig)
        with notes:
            st.subheader("What to notice")
            st.caption("A cluster near $20 (no internet) and a wide hump from $70 to $100.")

    with v3:
        chart, notes = st.columns([2, 1])
        with chart:
            # Scatter plot: relationship between two quantitative variables
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered, x="tenure", y="MonthlyCharges",
                            hue="InternetService", alpha=0.3, palette="colorblind", ax=ax)
            ax.set_title("Tenure vs. monthly charge")
            ax.set_xlabel("Tenure (months)")
            ax.set_ylabel("Monthly charge ($)")
            st.pyplot(fig)
        with notes:
            st.subheader("What to notice")
            st.caption("Bills separate into bands by internet type, not by tenure.")

    with v4:
        chart, notes = st.columns([2, 1])
        with chart:
            # Correlation heatmap (bonus)
            fig, ax = plt.subplots()
            sns.heatmap(filtered.select_dtypes("number").corr(), annot=True, ax=ax)
            ax.set_title("Correlation heatmap")
            st.pyplot(fig)
        with notes:
            st.subheader("What to notice")
            st.caption("Tenure and Monthly Charges are only weakly correlated. "
                       "tenure and tenure_years correlate at 1.0 because one is just the other divided by 12.")

    with v5:
        st.subheader("Average monthly charge by churn")
        st.dataframe(filtered.groupby("Churn")["MonthlyCharges"].mean())
        st.caption("Churn = No means the customer stayed; Churn = Yes means the customer left.")
        st.info("Customers who churned pay more per month on average than customers who stayed.")


# ============================================================
# CHUNK 7: the models + the Model tab
# ============================================================
# @st.cache_resource = same idea as cache_data, but for trained models
@st.cache_resource
def fit_models(data):
    y = data["MonthlyCharges"]            # the target: what we predict

    # ONE-HOT ENCODING: models only accept numbers, so turn InternetService
    # (DSL / Fiber optic / No) into 0/1 columns. drop_first=True drops DSL
    # (alphabetically first) so it becomes the reference level. We encode the
    # WHOLE df before splitting so train and test end up with the same columns.
    enc = pd.get_dummies(data, columns=["InternetService"], drop_first=True, dtype=int)
    X = enc[["tenure"] + [c for c in enc.columns if c.startswith("InternetService_")]]

    # 80% train / 20% test. random_state=42 makes the split the same every run.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Safety net: force test to have exactly the same columns as train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # MODEL 1, Constant: predict the training average for everyone.
    # RMSE = square root of the average of (actual - predicted) squared.
    const_pred = y_train.mean()
    rmse_const = ((y_test - const_pred) ** 2).mean() ** 0.5

    # MODEL 2, OLS WITHOUT the encoded feature (tenure only), for the comparison
    model_without = LinearRegression().fit(X_train[["tenure"]], y_train)
    rmse_without = mean_squared_error(y_test, model_without.predict(X_test[["tenure"]])) ** 0.5

    # MODEL 3, OLS WITH the encoded feature
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse_with = mean_squared_error(y_test, preds) ** 0.5

    # One coefficient per feature, labeled with the feature name
    coefs = pd.Series(model.coef_, index=X_train.columns)

    return {"rmse_const": rmse_const, "rmse_without": rmse_without,
            "rmse_with": rmse_with, "coefs": coefs,
            "y_test": y_test, "preds": preds,
            "n_train": len(X_train), "n_test": len(X_test)}


m = fit_models(df)

with tab3:
    st.header("Predicting Monthly Charges")
    st.markdown(
        "**Split:** 80% train / 20% test (`random_state=42`). "
        "**One-hot encoded feature:** `InternetService`. "
        "**Reference level dropped: DSL.**"
    )
    st.write("Train rows:", m["n_train"], "| Test rows:", m["n_test"])

    # Compare all three models side by side
    a, b, c = st.columns(3)
    a.metric("Constant model RMSE ($)", round(m["rmse_const"], 2))
    b.metric("OLS, tenure only RMSE ($)", round(m["rmse_without"], 2))
    c.metric("OLS + InternetService RMSE ($)", round(m["rmse_with"], 2))
    st.caption("Adding the one-hot encoded feature cut test error by more than half.")

    st.subheader("Coefficients")
    st.dataframe(m["coefs"])
    st.markdown(
        "- `InternetService_Fiber optic` is about 33.6: holding tenure constant, fiber optic "
        "customers are predicted to pay about **$33.60 more per month** than DSL customers.\n"
        "- `InternetService_No` is about -36.3: holding tenure constant, customers with no "
        "internet are predicted to pay about **$36.30 less per month** than DSL customers."
    )

    # Predicted vs. actual: points on the red line = perfect predictions
    st.subheader("Predicted vs. actual (test set)")
    fig, ax = plt.subplots()
    ax.scatter(m["y_test"], m["preds"], alpha=0.6)
    ax.plot([m["y_test"].min(), m["y_test"].max()],
            [m["y_test"].min(), m["y_test"].max()], color="red")
    ax.set_xlabel("Actual monthly charge ($)")
    ax.set_ylabel("Predicted monthly charge ($)")
    st.pyplot(fig)
    st.caption("Each dot is one customer from the test set. Closer to the red line = better prediction.")
    st.markdown(
        """
**How to read this plot**
- Each dot is one customer in the **test set** (the 20% of customers the model never saw while learning).
- The x-axis is the customer's **actual** monthly charge. The y-axis is what the model **predicted**.
- The red line is perfect prediction (predicted = actual). A dot **above** the line means the model
  guessed too high; a dot **below** it means the model guessed too low.

**What it shows**
- The dots form three rough bands, one per internet type: lowest is no internet, middle is DSL,
  top is fiber optic. The model mostly predicts a customer's bill from their internet type;
  tenure only nudges the prediction up a little (about $0.27 per month of tenure).
- Inside each band, actual bills vary a lot but predictions barely move. For DSL customers,
  actual bills range from about $24 to $93, but predictions only range from about $50 to $69.
  So the model cannot tell a cheap DSL customer from an expensive one. Something other than
  internet type and tenure (for example, add-on services) must explain that gap.
- At the high end, dots fall below the red line: the model **underpredicts** the most expensive
  customers. The top 20% of actual bills are underpredicted by about $8 on average.
- A typical prediction is off by about $11 (the test RMSE), compared with about $30 for the
  constant model that guesses the average for everyone.
        """
    )