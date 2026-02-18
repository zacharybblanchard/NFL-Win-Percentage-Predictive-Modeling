import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

#load the data
df = pd.read_csv("nfl_team_stats.csv")

#preview the first 5 rows
print(df.head())

#show basic info about the dataset
print(df.info())

#replace NaN ties with 0
df["ties"] = df["ties"].fillna(0)

# Create total games column
df["total_games"] = df["wins"] + df["losses"] + df["ties"]

#Create win percentage
df["win_percentage"] = df["wins"] / df["total_games"]

print(df[["team", "year", "win_percentage"]].head())

#select relevant columns for analysis
analysis_df=df[[
"win_loss_perc",
"points",
"points_opp",
    "points_diff",
    "total_yards",
    "turnovers",
    "pass_net_yds_per_att",
    "rush_yds_per_att",
    "turnover_pct",
    "score_pct"
]]

print(analysis_df.head())

#Correlation Matrix
corr = analysis_df.corr()

print(corr)

#Define Features (Predictors)
X = df[[
	"pass_net_yds_per_att",
	"turnover_pct",
	"score_pct",
	"total_yards",
	"points_opp"
]]

#Define Target Variable
y = df["win_loss_perc"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split using scaled data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


#Create model
model = LinearRegression()

# Perform 5-fold cross validation
cv_scores = cross_val_score(
    model,
    X_scaled,
    y,
    cv=5,
    scoring="r2"
)


print("Cross-Validation R2 Scores:", cv_scores)
print("Average CV R2:", cv_scores.mean())


#Train Model
model.fit(X_train, y_train)

#predict on test data
y_pred = model.predict(X_test)

#R-squared
r2 = r2_score(y_test, y_pred)

#RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R-squared", r2)
print("RMSE", rmse)

#Display feature importance


coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Standardized Coefficient": model.coef_
})

# Sort by absolute importance
coefficients["Abs_Value"] = coefficients["Standardized Coefficient"].abs()
coefficients = coefficients.sort_values("Abs_Value", ascending=False)

print(coefficients)

# Plot feature importance
plt.figure()
plt.bar(coefficients["Feature"], coefficients["Standardized Coefficient"])

plt.xticks(rotation=45)
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.title("Impact of Team Stats on Win Percentage")

plt.show()