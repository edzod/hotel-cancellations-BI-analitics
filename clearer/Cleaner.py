# ייבוא ספריות רלוונטיות
import pandas as pd
import numpy as np

# שלב 1: טעינת הנתונים (Extract + Load)
file_path = "Data/hotel_bookings.csv"
df = pd.read_csv(file_path)

# שלב 2: ניקוי נתונים (Cleaning)

# מחיקת עמודת company (כי רוב הערכים חסרים)
df.drop(columns=['company'], inplace=True)

#agent 0- הזמנה ישירות מהמלון
# מילוי ערכים חסרים ב-agent
df['agent'] = df['agent'].fillna(0).astype(int)

# מילוי ערכים חסרים ב-children
df['children'] = df['children'].fillna(0)

# מילוי ערכים חסרים ב-country
most_common_country = df['country'].mode()[0]
df['country'] = df['country'].fillna(most_common_country)



# שלב 3: המרת טיפוסי נתונים (Type Conversion)

# המרת תאריכים
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

# המרת מזהים/אינדיקטורים
df['agent'] = df['agent'].astype(int)
df['children'] = df['children'].astype(int)
df['adults'] = df['adults'].astype(int)
df['babies'] = df['babies'].astype(int)
df['is_canceled'] = df['is_canceled'].astype(bool)
df['is_repeated_guest'] = df['is_repeated_guest'].astype(bool)
df['required_car_parking_spaces'] = df['required_car_parking_spaces'].astype(bool)

# המרת קטגוריות
categorical_columns = [
    'hotel', 'meal', 'market_segment', 'distribution_channel',
    'reserved_room_type', 'assigned_room_type', 'deposit_type',
    'customer_type', 'reservation_status', 'country'
]
df[categorical_columns] = df[categorical_columns].astype('category')


# שלב 4: שמירת הקובץ הסופי
output_path = "Data/cleaned_hotel_bookings.csv"
df.to_csv(output_path, index=False)
