import os
import django
import csv
import pandas as pd
from django.db import models
from django.core.management.base import BaseCommand
from sklearn.linear_model import LinearRegression
import joblib
os.environ.setdefault("StartUpProfit/settings.py", "SatrtUpProfit/50_Startups.csv")
django.setup()

class Startup(models.Model):
    rd_spend = models.FloatField(verbose_name="R&D Spend")
    administration = models.FloatField(verbose_name="Administration")
    marketing_spend = models.FloatField(verbose_name="Marketing Spend")
    profit = models.FloatField(verbose_name="Profit")

    def __str__(self):
        return f"Startup({self.rd_spend}, {self.administration}, {self.marketing_spend}, {self.profit})"

# Load data from CSV file into the database
def load_data_from_csv(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Startup.objects.create(
                rd_spend=row['rd_spend'],
                administration=row['administration'],
                marketing_spend=row['marketing_spend'],
                profit=row['profit']
            )

# Train the multiple linear regression model
def train_and_save_model():
    # Load data from the database
    data = Startup.objects.all().values()
    df = pd.DataFrame(data)

    # Define features and target
    X = df[['rd_spend', 'administration', 'marketing_spend']]
    y = df['profit']

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, 'startups2.sav')
    print("Model saved as startups2.sav")

# Main execution
if __name__ == "__main__":
    # Path to your CSV file
    csv_path = 'D:/Projects/StartUpProfit/50_Startups.csv'

    # Load data into the database
    load_data_from_csv(csv_path)

    # Train and save the model
    train_and_save_model()
