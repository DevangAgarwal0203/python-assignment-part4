# part4_visualization_ml.py
# Student Performance Analysis & Prediction — Part 4

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# TASK 1 — Data Exploration with Pandas
# ─────────────────────────────────────────────

print("=" * 50)
print("TASK 1 — Data Exploration")
print("=" * 50)

df = pd.read_csv("students.csv")

print("\nFirst 5 rows:")
print(df.head())

print(f"\nShape : {df.shape[0]} rows × {df.shape[1]} columns")
print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe())

print("\nPass / Fail count:")
print(df["passed"].value_counts())

# average per subject split by pass/fail
subject_cols = ["math", "science", "english", "history", "pe"]

pass_avg = df[df["passed"] == 1][subject_cols].mean()
fail_avg = df[df["passed"] == 0][subject_cols].mean()

print("\nAverage scores — Passing students:")
print(pass_avg.round(2))

print("\nAverage scores — Failing students:")
print(fail_avg.round(2))

# student with highest overall average
df["temp_avg"] = df[subject_cols].mean(axis=1)
top_idx  = df["temp_avg"].idxmax()
top_name = df.loc[top_idx, "name"]
top_avg  = df.loc[top_idx, "temp_avg"]
print(f"\nTop student : {top_name} (avg = {top_avg:.2f})")

df.drop(columns=["temp_avg"], inplace=True)


# ─────────────────────────────────────────────
# TASK 2 — Data Visualization with Matplotlib
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("TASK 2 — Matplotlib Visualizations")
print("=" * 50)

# add avg_score column (used in scatter)
df["avg_score"] = df[subject_cols].mean(axis=1)

# Plot 1 — Bar chart: average score per subject
fig, ax = plt.subplots(figsize=(8, 5))
subject_means = df[subject_cols].mean()
ax.bar(subject_means.index, subject_means.values, color="steelblue", edgecolor="black")
ax.set_title("Average Score per Subject (All Students)")
ax.set_xlabel("Subject")
ax.set_ylabel("Average Score")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("plot1_bar.png")
plt.show()
print("Saved plot1_bar.png")

# Plot 2 — Histogram: math score distribution
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df["math"], bins=5, color="coral", edgecolor="black")
mean_math = df["math"].mean()
ax.axvline(mean_math, color="navy", linestyle="--", label=f"Mean = {mean_math:.1f}")
ax.set_title("Distribution of Math Scores")
ax.set_xlabel("Math Score")
ax.set_ylabel("Number of Students")
ax.legend()
plt.tight_layout()
plt.savefig("plot2_histogram.png")
plt.show()
print("Saved plot2_histogram.png")

# Plot 3 — Scatter: study hours vs avg score, coloured by passed
pass_df = df[df["passed"] == 1]
fail_df = df[df["passed"] == 0]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pass_df["study_hours_per_day"], pass_df["avg_score"],
           color="green", label="Pass", marker="o", s=80)
ax.scatter(fail_df["study_hours_per_day"], fail_df["avg_score"],
           color="red", label="Fail", marker="x", s=80)
ax.set_title("Study Hours vs Average Score")
ax.set_xlabel("Study Hours per Day")
ax.set_ylabel("Average Score")
ax.legend()
plt.tight_layout()
plt.savefig("plot3_scatter.png")
plt.show()
print("Saved plot3_scatter.png")

# Plot 4 — Box plot: attendance_pct for Pass vs Fail
pass_att = df[df["passed"] == 1]["attendance_pct"].tolist()
fail_att = df[df["passed"] == 0]["attendance_pct"].tolist()

fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([pass_att, fail_att], labels=["Pass", "Fail"])
ax.set_title("Attendance % Distribution by Pass/Fail")
ax.set_xlabel("Result")
ax.set_ylabel("Attendance %")
plt.tight_layout()
plt.savefig("plot4_boxplot.png")
plt.show()
print("Saved plot4_boxplot.png")

# Plot 5 — Line plot: math and science per student
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["name"], df["math"],    marker="o", label="Math",    linestyle="-")
ax.plot(df["name"], df["science"], marker="s", label="Science", linestyle="--")
ax.set_title("Math and Science Scores per Student")
ax.set_xlabel("Student")
ax.set_ylabel("Score")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plot5_line.png")
plt.show()
print("Saved plot5_line.png")


# ─────────────────────────────────────────────
# TASK 3 — Seaborn Visualizations
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("TASK 3 — Seaborn Visualizations")
print("=" * 50)

# Plot 6 — Bar: avg math and science split by passed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

sns.barplot(data=df, x="passed", y="math",    ax=ax1, palette="Blues")
ax1.set_title("Avg Math Score by Pass/Fail")
ax1.set_xlabel("Passed (0=Fail, 1=Pass)")
ax1.set_ylabel("Math Score")

sns.barplot(data=df, x="passed", y="science", ax=ax2, palette="Oranges")
ax2.set_title("Avg Science Score by Pass/Fail")
ax2.set_xlabel("Passed (0=Fail, 1=Pass)")
ax2.set_ylabel("Science Score")

plt.tight_layout()
plt.savefig("plot6_seaborn_bar.png")
plt.show()
print("Saved plot6_seaborn_bar.png")

# Plot 7 — Scatter + regression: attendance_pct vs avg_score by passed
fig, ax = plt.subplots(figsize=(8, 5))

sns.regplot(data=df[df["passed"] == 1], x="attendance_pct", y="avg_score",
            ax=ax, label="Pass", color="green", scatter_kws={"s": 60})
sns.regplot(data=df[df["passed"] == 0], x="attendance_pct", y="avg_score",
            ax=ax, label="Fail", color="red",   scatter_kws={"s": 60})

ax.set_title("Attendance % vs Avg Score (with Regression)")
ax.set_xlabel("Attendance %")
ax.set_ylabel("Average Score")
ax.legend()
plt.tight_layout()
plt.savefig("plot7_seaborn_scatter.png")
plt.show()
print("Saved plot7_seaborn_scatter.png")

# Seaborn vs Matplotlib comparison (student observation):
# Seaborn was noticeably quicker for the bar chart split by category since you
# just pass x='passed' and it groups automatically. The regplot was also nice —
# one call added both the scatter points and the regression line. Matplotlib
# needed more manual steps for the same things, but it gave more control over
# exactly how each plot looked, which was useful for the box plot and line plot.


# ─────────────────────────────────────────────
# TASK 4 — Machine Learning with scikit-learn
# ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("TASK 4 — Machine Learning")
print("=" * 50)

# Step 1 — Prepare data
feature_cols = ["math", "science", "english", "history", "pe",
                "attendance_pct", "study_hours_per_day"]

X = df[feature_cols]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Step 2 — Train
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
print(f"\nTraining accuracy : {train_acc * 100:.2f}%")

# Step 3 — Evaluate
y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy     : {test_acc * 100:.2f}%")

print("\nPer-student predictions on test set:")
test_names = df.loc[X_test.index, "name"]
for name, actual, pred in zip(test_names, y_test, y_pred):
    result = "✅ Correct" if actual == pred else "❌ Wrong"
    print(f"  {name:<10}  Actual: {actual}  Predicted: {pred}  {result}")

# Step 4 — Feature importance
coefficients = model.coef_[0]
feat_imp = list(zip(feature_cols, coefficients))
feat_imp.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nFeature coefficients (sorted by |value|):")
for feat, coef in feat_imp:
    print(f"  {feat:<25} {coef:+.4f}")

# horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 5))
feats  = [f for f, _ in feat_imp]
coefs  = [c for _, c in feat_imp]
colors = ["green" if c > 0 else "red" for c in coefs]

ax.barh(feats, coefs, color=colors, edgecolor="black")
ax.set_title("Logistic Regression Feature Coefficients")
ax.set_xlabel("Coefficient Value")
ax.set_ylabel("Feature")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("plot8_feature_importance.png")
plt.show()
print("Saved plot8_feature_importance.png")

# Step 5 — Bonus: predict for a new student
print("\n--- Bonus: New Student Prediction ---")
new_student = [[75, 70, 68, 65, 80, 82, 3.2]]

new_scaled  = scaler.transform(new_student)
prediction  = model.predict(new_scaled)[0]
probability = model.predict_proba(new_scaled)[0]

result_label = "Pass ✅" if prediction == 1 else "Fail ❌"
print(f"Prediction  : {result_label}")
print(f"Probability : Fail = {probability[0]:.4f},  Pass = {probability[1]:.4f}")
