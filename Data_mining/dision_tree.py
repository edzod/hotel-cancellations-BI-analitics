import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# שלב 1: טעינת הנתונים
fact_data = pd.read_csv('Fact_Booking (1).csv')

# הצצה לנתונים
print(fact_data.head())

# שלב 2: בחירת עמודת היעד והמאפיינים
target = 'reservation_status'
features = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'deposit_type', 'customer_type', 'adr', 'total_of_special_requests', 'booking_changes']

# שלב 3: טיפול בערכים חסרים
data = fact_data.dropna()

# שלב 4: המרת משתנים קטגוריאליים למספריים
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])
data['deposit_type'] = label_encoder.fit_transform(data['deposit_type'])
data['customer_type'] = label_encoder.fit_transform(data['customer_type'])

# שלב 5: חלוקה ל-Train/Test
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# שלב 6: בניית Decision Tree
clf_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_dt.fit(X_train, y_train)
y_pred_dt = clf_dt.predict(X_test)

# בניית Random Forest
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# בניית K-Nearest Neighbors
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)

# שלב 7: הערכת המודלים
def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

evaluate_model(y_test, y_pred_dt, 'Decision Tree')
evaluate_model(y_test, y_pred_rf, 'Random Forest')
evaluate_model(y_test, y_pred_knn, 'KNN')

# שלב 8: Feature Importance עבור Decision Tree
feature_importances = clf_dt.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance - Decision Tree:")
print(feature_importance_df)

# שלב 9: ROC Curve עבור Decision Tree
y_prob_dt = clf_dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt, pos_label=1)
plt.plot(fpr_dt, tpr_dt, label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend()
plt.grid()
plt.show()
