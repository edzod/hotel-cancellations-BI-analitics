import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# שלב 1: טעינת הנתונים
data = pd.read_csv('Fact_Booking (1).csv')

# שלב 2: בחירת מאפיינים
features = ['previous_cancellations', 'lead_time']
clustering_data = data[features].dropna()

# שלב 3: נרמול הנתונים
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# שלב 4: יישום K-Means עם 3 אשכולות
kmeans = KMeans(n_clusters=3, random_state=42)
clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)

# שלב 5: ויזואליזציה
plt.figure(figsize=(12, 6))
plt.scatter(clustering_data['lead_time'], clustering_data['previous_cancellations'], c=clustering_data['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Lead Time')
plt.ylabel('Previous Cancellations')
plt.title('K-Means Clustering - Lead Time vs Previous Cancellations')
plt.grid(alpha=0.3)
plt.colorbar(label='Cluster')
plt.show()

#חישובים סטטיסטיים
# חישוב סיכום סטטיסטי לפי אשכול
summary = clustering_data.groupby('Cluster').agg({
    'lead_time': 'mean',
    'previous_cancellations': 'mean'
})

# הוספת ספירת לקוחות בכל אשכול
summary['Cluster Size'] = clustering_data['Cluster'].value_counts().sort_index()

# סידור העמודות מחדש
summary = summary[['Cluster Size', 'lead_time', 'previous_cancellations']]

# עיגול תוצאות לנוחות
summary = summary.round(2)

print(summary)
