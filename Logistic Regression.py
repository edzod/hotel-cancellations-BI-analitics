import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# שלב 1: טעינת הנתונים
data = pd.read_csv('Fact_Booking (1).csv')

# שלב 2: בחירת מאפיינים ועמודת היעד
features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'deposit_type', 'customer_type', 'adr', 'total_of_special_requests', 'booking_changes']
target = 'is_canceled'

# שלב 3: הסרת ערכים חסרים
data = data.dropna()

# שלב 4: המרת משתנים קטגוריאליים למספריים
label_encoder = LabelEncoder()
data['deposit_type'] = label_encoder.fit_transform(data['deposit_type'])
data['customer_type'] = label_encoder.fit_transform(data['customer_type'])
data[target] = label_encoder.fit_transform(data[target])

# שלב 5: חלוקת הנתונים
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# שלב 6: בניית המודל Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# שלב 7: מדדי הערכה
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# שלב 8: גרף ROC Curve
y_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid()
plt.show()