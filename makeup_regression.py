# %%

# ============================================================================
# Regression Overview — In-Class Live Coding Example
# Dataset: Facebook Performance Metrics (UCI ML Repo ID 368)
# Topics:
#   1. Kernel Density Plot
#   2. Dummy Variables
#   3. Regression Without an Intercept in sklearn
#   4. Multivariate Regression
#   5. Log and arcsinh Transformations
#   6. Dummy Variable Trap (drop one level as reference)
#   7. Polynomial Features from sklearn
#   8. True vs. Predicted Plot with Train/Test Split
# =============================================================================
# %%


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# %%
# --- Load Dataset ---
facebook_metrics = fetch_ucirepo(id=368)

X = facebook_metrics.data.features.copy()
y = facebook_metrics.data.targets.copy()

print(facebook_metrics.metadata)
print(facebook_metrics.variables)

# %%
# Combine for easier exploration
df = pd.concat([X, y], axis=1)
df.info()

# %%
# Drop rows where target (Total Interactions) is missing
df = df.dropna(subset=['Total Interactions'])
print(df.shape)

# =============================================================================
# SECTION 1: Kernel Density Plot
# =============================================================================
# A kernel density plot (KDE) is a smoothed version of a histogram.
# It estimates the probability density function of a continuous variable.
# Use it to understand the shape and spread of a distribution before modeling.

# %%
# KDE of the target variable: Total Interactions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['Total Interactions'].plot.kde(ax=axes[0], color='steelblue')
axes[0].set_title('KDE: Total Interactions (raw)')
axes[0].set_xlabel('Total Interactions')

# The raw distribution is heavily right-skewed — a common problem in regression.
# Let's also look at Page total likes
df['Page total likes'].plot.kde(ax=axes[1], color='coral')
axes[1].set_title('KDE: Page Total Likes (raw)')
axes[1].set_xlabel('Page Total Likes')

plt.tight_layout()
plt.show()

# KEY POINT: Skewed distributions can violate regression assumptions.
# We'll address this with log/arcsinh transformations in Section 5.

# =============================================================================
# SECTION 2: Dummy Variables
# =============================================================================
# Categorical variables cannot be fed directly into sklearn regression models.
# We convert them to binary (0/1) indicator columns — called dummy variables.
# pd.get_dummies() does this automatically.

# %%
print(df['Type'].value_counts())   # Photo, Status, Link, Video
print(df['Category'].value_counts())

# %%
# One-hot encode 'Type' — we get one binary column per category level
type_dummies = pd.get_dummies(df['Type'], prefix='type')
print(type_dummies.head(10))
# Each row gets a 1 in the column matching its post type, 0 everywhere else.

# =============================================================================
# SECTION 3: Regression WITHOUT an Intercept in sklearn
# =============================================================================
# By default, LinearRegression fits: y = b0 + b1*x1 + b2*x2 + ...
# Setting fit_intercept=False removes b0, forcing the line through the origin:
#   y = b1*x1 + b2*x2 + ...
# This is rarely appropriate unless theory demands it, but it's useful to know.

# %%
# Simple example: predict Total Interactions from Page total likes
X_simple = df[['Page total likes']].fillna(df['Page total likes'].median())
y_target = df['Total Interactions']

# With intercept (default)
model_with    = LinearRegression(fit_intercept=True).fit(X_simple, y_target)
# Without intercept
model_without = LinearRegression(fit_intercept=False).fit(X_simple, y_target)

print(f"WITH intercept    — coef: {model_with.coef_[0]:.4f},  intercept: {model_with.intercept_:.2f}")
print(f"WITHOUT intercept — coef: {model_without.coef_[0]:.4f},  intercept: {model_without.intercept_:.2f}")

# KEY POINT: Unless your domain knowledge justifies it, always keep the intercept.
# Forcing through the origin biases the slope estimate when y != 0 at x=0.

# =============================================================================
# SECTION 4: Multivariate Regression
# =============================================================================
# Simple regression: one predictor     → y = b0 + b1*x1
# Multivariate regression: many predictors → y = b0 + b1*x1 + b2*x2 + ... + bn*xn
#
# Each coefficient tells you the expected change in y for a 1-unit change in
# that predictor, HOLDING ALL OTHER PREDICTORS CONSTANT (ceteris paribus).
# This is the key advantage over simple regression — we can isolate effects.

# %%
# Select a handful of numeric features
numeric_features = ['Page total likes', 'Post Month', 'Post Weekday', 'Post Hour', 'Paid']

df_mv = df[numeric_features + ['Total Interactions']].dropna()

X_mv = df_mv[numeric_features]
y_mv = df_mv['Total Interactions']

model_mv = LinearRegression().fit(X_mv, y_mv)

coef_df = pd.DataFrame({'Feature': numeric_features, 'Coefficient': model_mv.coef_})
print(coef_df.to_string(index=False))
print(f"\nIntercept: {model_mv.intercept_:.2f}")
print(f"R²: {model_mv.score(X_mv, y_mv):.4f}")

# KEY POINT: R² tells us the fraction of variance in y explained by the model.
# A low R² here suggests the linear numeric features alone are not very informative.

# =============================================================================
# SECTION 5: Log and arcsinh Transformations of Feature Variables
# =============================================================================
# Why transform?
#   - Skewed predictors compress extreme values, improving linearity
#   - Reduces the influence of outliers
#   - Can improve model fit and residual normality
#
# log(x):    works only for strictly positive values (x > 0)
# arcsinh(x): works for zero and negative values — a generalization of log
#             arcsinh(x) ≈ log(2x) for large x, but handles 0s gracefully

# %%
# Look at the raw distribution of Page total likes
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df['Page total likes'].plot.kde(ax=axes[0], color='steelblue')
axes[0].set_title('Raw: Page Total Likes')

np.log(df['Page total likes'] + 1).plot.kde(ax=axes[1], color='green')
axes[1].set_title('log(Page Total Likes + 1)')

np.arcsinh(df['Page total likes']).plot.kde(ax=axes[2], color='purple')
axes[2].set_title('arcsinh(Page Total Likes)')

plt.tight_layout()
plt.show()

# %%
# Apply log and arcsinh transformations and add to dataframe
df['log_page_likes']     = np.log(df['Page total likes'] + 1)
df['arcsinh_page_likes'] = np.arcsinh(df['Page total likes'])
df['log_interactions']   = np.log(df['Total Interactions'] + 1)

# Compare correlation with target before and after transformation
print("Correlation with Total Interactions:")
print(f"  Raw Page Likes:     {df['Page total likes'].corr(df['Total Interactions']):.4f}")
print(f"  Log Page Likes:     {df['log_page_likes'].corr(df['Total Interactions']):.4f}")
print(f"  arcsinh Page Likes: {df['arcsinh_page_likes'].corr(df['Total Interactions']):.4f}")

# =============================================================================
# SECTION 6: Dummy Variable Trap — Drop One Level as Reference
# =============================================================================
# When you create k dummies for a k-level categorical variable and include ALL
# of them in a model WITH an intercept, you create perfect multicollinearity:
#   type_Photo + type_Status + type_Link + type_Video = 1 (always)
# This is the DUMMY VARIABLE TRAP — the model matrix becomes singular.
#
# Fix: drop one level (the "reference" category). Its effect is captured by
# the intercept. Coefficients on remaining dummies are interpreted RELATIVE to
# the dropped reference level.

# %%
# Wrong: include all levels (trap)
dummies_all = pd.get_dummies(df['Type'], prefix='type')
print("All dummy columns:", dummies_all.columns.tolist())
print("Sum across a row (always 1):\n", dummies_all.head(3).sum(axis=1).values)

# %%
# Correct: drop_first=True drops the first alphabetical level (Link → reference)
dummies_ref = pd.get_dummies(df['Type'], prefix='type', drop_first=True)
print("\nDummies after drop_first:", dummies_ref.columns.tolist())
print("'Link' is the reference — its effect is absorbed into the intercept.")
print(dummies_ref.head())

# Interpretation example:
# type_Photo coefficient = expected difference in Total Interactions for
# a Photo post vs. a Link post, holding everything else constant.

# =============================================================================
# SECTION 7: Polynomial Features from sklearn
# =============================================================================
# Linear regression assumes a straight-line relationship between X and y.
# If the true relationship is curved, we can add polynomial terms:
#   y = b0 + b1*x + b2*x² + b3*x³ + ...
#
# PolynomialFeatures() generates all polynomial and interaction terms
# up to the specified degree automatically.

# %%
from sklearn.preprocessing import PolynomialFeatures

# Use a single feature to illustrate the polynomial expansion clearly
X_poly_base = df[['log_page_likes']].dropna()
y_poly      = df.loc[X_poly_base.index, 'Total Interactions']

# Degree 2: adds x and x²
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_poly_base)

print("Original shape:", X_poly_base.shape)
print("Polynomial (degree=2) shape:", X_poly.shape)
print("Feature names:", poly.get_feature_names_out())

# %%
# Fit linear, degree-2, and degree-3 models on the same feature
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly_base, y_poly, test_size=0.2, random_state=42
)

results = {}
for degree in [1, 2, 3]:
    pf  = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr = pf.fit_transform(X_train_p)
    Xte = pf.transform(X_test_p)
    m   = LinearRegression().fit(Xtr, y_train_p)
    r2  = m.score(Xte, y_test_p)
    results[f'degree_{degree}'] = r2
    print(f"Degree {degree}  |  Test R²: {r2:.4f}")

# KEY POINT: Higher degree is not always better — watch for overfitting.
# Use train/test split (or cross-validation) to compare out-of-sample performance.

# =============================================================================
# SECTION 8: True vs. Predicted Values Plot with Train/Test Split
# =============================================================================
# The standard sklearn workflow:
#   1. Split data into train and test sets
#   2. Fit the model on training data only
#   3. Predict on test data
#   4. Evaluate on test data (unseen by the model)
#
# A true vs. predicted scatter plot is a quick visual diagnostic:
#   - Perfect model → points fall on the 45-degree line (y = x)
#   - Systematic deviations → bias or missing non-linearity

# %%
# Build a richer feature set using transformations and dummies
df_model = df[['log_page_likes', 'arcsinh_page_likes',
               'Post Month', 'Post Weekday', 'Post Hour', 'Paid',
               'Type', 'Total Interactions']].dropna()

# Dummy encode Type with reference level (drop_first)
type_enc = pd.get_dummies(df_model['Type'], prefix='type', drop_first=True)

X_full = pd.concat([
    df_model[['log_page_likes', 'arcsinh_page_likes',
              'Post Month', 'Post Weekday', 'Post Hour', 'Paid']],
    type_enc
], axis=1)

y_full = df_model['Total Interactions']

# %%
# Train / Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# Fit on training data only
final_model = LinearRegression().fit(X_train, y_train)

# Predict on test data
y_pred = final_model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"\nTest RMSE: {rmse:.2f}")
print(f"Test R²:   {r2:.4f}")

# %%
# True vs. Predicted Plot
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='steelblue',
           facecolors='none', linewidth=0.8, label='Predictions')

# 45-degree reference line (perfect predictions)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val],
        color='red', linestyle='--', linewidth=1.5, label='Perfect Fit')

ax.set_xlabel('True Total Interactions', fontsize=12)
ax.set_ylabel('Predicted Total Interactions', fontsize=12)
ax.set_title(f'True vs. Predicted — Test Set\nRMSE={rmse:.1f}  R²={r2:.3f}', fontsize=13)
ax.legend()
plt.tight_layout()
plt.show()

# KEY POINT: Points scattered around the red line is good.
# A fan shape (heteroskedasticity) or curve suggests a model limitation.
# Here, extreme values are hard to predict — common with social media data.

# %%
# Coefficient table for the final model
coef_final = pd.DataFrame({
    'Feature': X_full.columns,
    'Coefficient': final_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFinal Model Coefficients (sorted by magnitude):")
print(coef_final.to_string(index=False))
print(f"\nIntercept: {final_model.intercept_:.2f}")
