import datetime

import requests


class Intervals:
    """ """

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
        if type(start_date) is not datetime.date:
            raise TypeError("datetime required")

        params = {}

        if end_date is not None:
            if type(end_date) is not datetime.date:
                raise TypeError("datetime required")
            end_date = end_date + datetime.timedelta(days=1)
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

    def activities_csv(self):
        """
        Returns all your activities formatted in CSV

        :return: Text data in CSV format
        :rtype: str
        """
        url = "{}/api/v1/athlete/{}/activities.csv".format(
            Intervals.BASE_URL, self.athlete_id
        )
        res = self._make_request("get", url)
        return res.text

    def activity(self, activity_id):
        """ """
        url = "{}/api/v1/activity/{}".format(Intervals.BASE_URL, activity_id)
        res = self._make_request("get", url)
        return res.json()
        # return Activity(**res.json())

    def athlete(self, athlete_id):
        """ """
        url = "{}/api/v1/athlete/{}".format(Intervals.BASE_URL, athlete_id)
        res = self._make_request("get", url)
        fields = res.json()
        ride = run = swim = other = {}
        for sport in fields["sportSettings"]:
            if "Ride" in sport["types"]:
                ride = sport
                print("Ride", type(sport))
            if "Run" in sport["types"]:
                run = sport
                print("Run")
            if "Swim" in sport["types"]:
                swim = sport
                print("Swim")
            if "Other" in sport["types"]:
                other = sport
                print("Other")
        return ride, run, swim, other

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

    def wellness(self, start_date, end_date=None):
        """ """
        if type(start_date) is not datetime.date:
            raise TypeError("datetime required")

        params = {}

        if end_date is not None:
            if type(end_date) is not datetime.date:
                raise TypeError("datetime required")

            params["oldest"] = start_date.isoformat()
            params["newest"] = end_date.isoformat()
            url = "{}/api/v1/athlete/{}/wellness".format(
                Intervals.BASE_URL, self.athlete_id
            )
        else:
            url = "{}/api/v1/athlete/{}/wellness/{}".format(
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

    def workouts(self):
        """ """
        url = "{}/api/v1/athlete/{}/workouts".format(
            Intervals.BASE_URL, self.athlete_id
        )

        res = self._make_request("get", url)
        j = res.json()
        if type(j) is list:
            result = []
            for item in j:
                result.append(item)
            return result

        raise TypeError("Unexpected result from server")

    def workout(self, workout_id):
        """ """
        url = "{}/api/v1/athlete/{}/workouts/{}".format(
            Intervals.BASE_URL, self.athlete_id, workout_id
        )

        res = self._make_request("get", url)
        return res.json()

    def power_curve(
        self,
        newest=datetime.datetime.now(),
        curves="90d",
        type="Ride",
        include_ranks=False,
        sub_max_efforts=0,
        filters='[{"field_id": "type", "value": ["Ride", "VirtualRide"]}]',
    ):
        """ """
        url = f"{self.BASE_URL}/api/v1/athlete/{self.athlete_id}/power-curves"
        params = {
            "curves": curves,
            "type": type,
            "includeRanks": include_ranks,
            "subMaxEfforts": f"{sub_max_efforts}",
            "filters": filters,
            "newest": newest.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        res = self._make_request("get", url, params=params)
        return res.json()
