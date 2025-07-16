import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from typing import Sequence
import urllib3
import sqlite3
import geopandas as gpd
import plotly.express as px

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---- API fetch functions ----
def get_voyage_history(username: str, password: str, imo: Sequence[int], from_date: datetime, to_date: datetime):
    imo_str = ','.join(map(str, imo))
    url = f"https://position.stratumfive.com/ais/staticvoyage-by-imo/{imo_str}/{from_date.isoformat()}/{to_date.isoformat()}"
    response = requests.get(url, auth=requests.auth.HTTPBasicAuth(username, password), verify=False)
    response.raise_for_status()
    return response.json()

def get_ais_positions(username: str, password: str, mmsis: Sequence[int], from_date: datetime, to_date: datetime, sixhourly: str = 'true'):
    mmsi_str = ','.join(map(str, mmsis))
    url = f"https://position.stratumfive.com/ais/positions/{mmsi_str}/{from_date.isoformat()}/{to_date.isoformat()}?is6hourly={sixhourly}"
    response = requests.get(url, auth=requests.auth.HTTPBasicAuth(username, password), verify=False)
    response.raise_for_status()
    return response.json()

def detect_mmsi_changes(voyage_data, end_date: str):
    mmsi_values = [entry['mmsi'] for entry in voyage_data]
    mmsis = list(set(mmsi_values))
    previous_mmsi = None
    timestamp_changes = []

    for entry in voyage_data:
        current_mmsi = entry['mmsi']
        timestamp = entry['timestamp']
        if current_mmsi != previous_mmsi:
            if previous_mmsi is not None:
                timestamp_changes.append((previous_mmsi, current_mmsi, timestamp))
        previous_mmsi = current_mmsi

    if not timestamp_changes:
        timestamp_changes.append((mmsis[0], mmsis[0], end_date))
    elif len(timestamp_changes) > 1:
        source = timestamp_changes[-1][1]
        sourcetime = timestamp_changes[-1][2]
        for i in range(len(timestamp_changes) - 1, -1, -1):
            if timestamp_changes[i][0] in [row[1] for row in timestamp_changes[:i]]:
                timestamp_changes.pop(i)
        timestamp_changes = [(timestamp_changes[0][0], source, sourcetime)]

    return timestamp_changes

def fetch_and_combine_ais(username, password, timestamp_changes, start, end, sixhourly):
    df_combined = pd.DataFrame()
    start_dt = datetime.fromisoformat(start + 'T00:00:00')
    end_dt = datetime.fromisoformat(end + 'T00:00:00')

    for mmsi in [timestamp_changes[0][0], timestamp_changes[0][1]]:
        ais_data = get_ais_positions(username, password, [mmsi], start_dt, end_dt, sixhourly)
        df = pd.DataFrame.from_dict(ais_data)
        if df.empty:
            continue

        df['DateTime'] = pd.to_datetime(df['timestamp'])
        df['latitude'] = df['lat']
        df['longitude'] = df['lon']
        df['speed'] = df['sogKts']
        df = df[['DateTime', 'speed', 'latitude', 'longitude']]

        switch_time = datetime.fromisoformat(timestamp_changes[0][2])
        if mmsi == timestamp_changes[0][0]:
            df = df[df['DateTime'] < switch_time]
        else:
            df = df[df['DateTime'] > switch_time]

        df_combined = pd.concat([df_combined, df])

    df_cleaned = df_combined.drop_duplicates(subset='DateTime').reset_index(drop=True)
    return df_cleaned


# --- Streamlit UI ---
st.title("🚢 Combined Voyage and AIS Dashboard")

username = st.secrets["username"]
password = st.secrets["password"]

imo_input = st.text_input("IMO number(s) (comma separated)", value="9770634")
start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
end_date = st.date_input("End Date", value=datetime(2024, 1, 10))
sixhourly = st.selectbox("6-Hourly data?", options=["true", "false"], index=0)

if st.button("Fetch Data"):
    with st.spinner("Fetching voyage and AIS data..."):
        try:
            imo_list = list(map(int, imo_input.split(',')))
            voyage_data = get_voyage_history(username, password, imo_list, start_date, end_date)
            timestamp_changes = detect_mmsi_changes(voyage_data, end_date.isoformat())

            ais_df = fetch_and_combine_ais(
                username, password,
                timestamp_changes,
                start_date.isoformat(),
                end_date.isoformat(),
                sixhourly
            )

            if ais_df.empty:
                st.warning("No AIS position data found for the selected criteria.")
            else:
                st.success("AIS Data fetched successfully!")
                st.subheader("AIS Positions")
                st.dataframe(ais_df)

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"Unexpected error during API call: {e}")

    # ---- SQLite vessel info ----
    try:
        imo_for_db = imo_list[0]
        with sqlite3.connect("my_sqlite.db") as cnn:
            query = f"SELECT * FROM vesselInfo WHERE LRIMOShipNo = {imo_for_db};"
            dfVesselInfo = pd.read_sql(query, cnn)

        if not dfVesselInfo.empty:
            st.subheader("📄 Vessel Information from Local DB")
            st.dataframe(dfVesselInfo)
        else:
            st.warning(f"No vessel info found in local DB for IMO {imo_for_db}")

    except Exception as e:
        st.error(f"SQLite DB error: {e}")
