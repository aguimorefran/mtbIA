# mtb-IA

## Route completion time



### Data retrieval 

Execute `python fetch_data.py` to download and prepare the data from Intervals ICU. The data will be downloaded to `activity_data.csv` file.

The script takes the following arguments:

- `start_date`: The start date of the data retrieval in format dd-mm-yyyy. Default is 02/12/2023.
- `ATHLETE_ID`: The athlete ID to retrieve the data from. Default is 1.
- `API_KEY`: The API key to access the data. Default is '1234'.

If ATHLETE_ID and API_KEY are not provided, the script will try to read them from the environment variables `ATHLETE_ID` and `API_KEY` respectively.