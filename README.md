# Hotel Bookings – BI Analytics (Capstone)

**Grade:** 93/100 (**Top 1/59**)  
**Tools:** Python (ETL/ML), SQL, Tableau, Excel

This repository contains my **Business Intelligence capstone** on the **Hotel Bookings dataset**.  
The project includes **data cleaning (ETL), star schema design, KPI suite, machine learning experiments, and 10 Tableau dashboards** with drilldowns & filters, providing insights on **seasonality and cancellation risk**.

---

## Repository Structure
- "Dimentions/" - Star schema CSVs (dimensions & fact)
- "cleaner/" - Python cleaning script & cleaned dataset
- "Data_mining/" - Logistic Regression, K-Means clustering, Decision Tree
- "assets/" - Tableau packaged workbook (`.twbx`) + visuals
- "Presentations/" - Written report & presentation
- "README.md" - This file

---

## Star Schema
- **Fact table:** "Fact_Booking.csv"
- **Dimensions:** "Dim_Agent", "Dim_Customer", "Dim_Date", "Dim_Hotel",  
  "Dim_MarketSegment", "Dim_Meal", "Dim_Order_Status", "Dim_RoomType"

---

## KPIs
- ADR (Average Daily Rate)  
- Cancellation rate  
- Lead time  
- (Optional) Occupancy & RevPAR if hotel capacity is modeled  

---

## Dashboards (Tableau)
1. Executive overview  
2. Seasonality trends  
3. Market mix (channels, country, segment)  
4. Cancellation analysis  
5. Length of stay patterns  
6. ADR price bands  
7. Geography map  
8. Operational arrivals/departures  
9. Agent/company performance  
10. Drill-through detail with filters  

---

## Machine Learning Extensions
In addition to BI analytics:
- **Logistic Regression** – predicting cancellations  
- **Decision Tree** – interpretable classification  
- **K-Means Clustering** – customer segmentation  

---

## Key Insights
- Cancellations increase with long lead times and non-deposit bookings.  
- Resort vs city hotels show different seasonality and ADR patterns.  
- Online TA bookings contribute the most to cancellation risk.  

---

📊 **Final grade:** 93/100 – top 1 out of 59 projects
