from dotenv import load_dotenv
import os
import pandas
import requests
from datetime import datetime

load_dotenv()

ICU_ATHLETE_ID = os.getenv("ICU_ATHLETE_ID")
ICU_API_KEY = os.getenv("ICU_API_KEY")

# https://codeberg.org/tmllull/intervalsicu-to-influxdb/src/branch/main/src/intervalsicu_to_influxdb/clients/intervals_client.py
class Intervals:
    BASE_URL = "https://intervals.icu"

    def __init__(self, athlete_id, api_key, session=None):
        """ """
        self.athlete_id = athlete_id
        self.password = api_key
        self.session = session

    def _get_session(self):
        if self.session is not None:
            return self.session

        self.session = requests.Session()

        self.session.auth = ("API_KEY", self.password)
        return self.session

    def _make_request(self, method, url, params=None):
        session = self._get_session()

        res = session.request(method, url, params=params)

        if res.status_code != 200:
            raise Exception("Error on request:" + str(res))

        return res

    def activities(self, start_date, end_date=None):
        """
        Returns all your activities formatted in CSV

        :return: Text data in CSV format
        :rtype: str
        """

        params = {}

        if end_date is not None:
            params["oldest"] = start_date.isoformat()
            params["newest"] = end_date.isoformat()
            url = "{}/api/v1/athlete/{}/activities".format(
                Intervals.BASE_URL, self.athlete_id
            )
        else:
            url = "{}/api/v1/athlete/{}/activities/{}".format(
                Intervals.BASE_URL, self.athlete_id, start_date.isoformat()
            )
        res = self._make_request("get", url, params)
        j = res.json()
        if type(j) is list:
            result = []
            for item in j:
                result.append(item)
            return result

        return j

    def activity(self, activity_id):
        """ """
        url = "{}/api/v1/activity/{}".format(Intervals.BASE_URL, activity_id)
        res = self._make_request("get", url)
        return res.json()
        # return Activity(**res.json())

    def activitiy_streams(self, activity_id):
        """
        Returns all your activities formatted in CSV

        :return: Text data in CSV format
        :rtype: str
        """
        url = "{}/api/v1/activity/{}/streams".format(Intervals.BASE_URL, activity_id)
        res = self._make_request("get", url)
        j = res.json()
        time = []
        watts = []
        cadence = []
        heartrate = []
        distance = []
        altitude = []
        latlng = []
        velocity_smooth = []
        temp = []
        torque = []
        respiration = []
        for stream in j:
            try:
                if stream["type"] == "time":
                    time = stream
                elif stream["type"] == "watts":
                    watts = stream
                elif stream["type"] == "cadence":
                    cadence = stream
                elif stream["type"] == "heartrate":
                    heartrate = stream
                elif stream["type"] == "distance":
                    distance = stream
                elif stream["type"] == "altitude":
                    altitude = stream
                elif stream["type"] == "latlng":
                    latlng = stream
                elif stream["type"] == "velocity_smooth":
                    velocity_smooth = stream
                elif stream["type"] == "temp":
                    temp = stream
                elif stream["type"] == "torque":
                    torque = stream
                elif stream["type"] == "respiration":
                    respiration = stream
            except Exception as e:
                print("Error on activity", activity_id, ":", e)

        return (
            time,
            watts,
            cadence,
            heartrate,
            distance,
            altitude,
            latlng,
            velocity_smooth,
            temp,
            torque,
            respiration,
        )



# test
newest_date = datetime(2023, 1, 1).date()
oldest_date = datetime(2024, 1, 31).date()
intervals = Intervals(ICU_ATHLETE_ID, ICU_API_KEY)
intervals.export_activities(newest_date, oldest_date, "icu_activities")
