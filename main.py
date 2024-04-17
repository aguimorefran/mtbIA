from dotenv import load_dotenv
import os
import datetime
load_dotenv()

from intervals_icu import Intervals

# Test the class
ICU_ATHLETE_ID = os.getenv("ICU_ATHLETE_ID")
ICU_API_KEY = os.getenv("ICU_API_KEY")

icu = Intervals(athlete_id=ICU_ATHLETE_ID, api_key=ICU_API_KEY)
newest_date = "2024-04-01"
oldes_date = "2024-01-01"
newest = datetime.datetime.strptime(newest_date, "%Y-%m-%d").date()
oldest = datetime.datetime.strptime(oldes_date, "%Y-%m-%d").date()

# Get all activities
activities = icu.activities(oldest, newest)
activities_df = icu.activities_pandas()
print("Activities:", activities_df)

# Save the first activity to a file
activity = activities_df.iloc[0]

# Save the first activity to a file
activity.to_csv("activity.csv")
