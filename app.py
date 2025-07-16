import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from typing import Sequence
import urllib3
import sqlite3
import geopandas as gpd
import plotly.express as px
import os 
from shapely.geometry import Point

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
    df = df_cleaned[df_cleaned["speed"] < 24]
    df = df.drop_duplicates(subset='DateTime').sort_values('DateTime').reset_index(drop=True)
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

            df_ais = fetch_and_combine_ais(
                username, password,
                timestamp_changes,
                start_date.isoformat(),
                end_date.isoformat(),
                sixhourly
            )
            if df_ais.empty:
                st.warning("No AIS position data found for the selected criteria.")
            else:
                st.success("AIS Data fetched successfully!")
                st.subheader("AIS Positions")
                st.dataframe(df_ais)

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

                # ---- LME Shapefile and Excel Info ----
                try:
                    LMEPolygon = "LMEPolygon1/LMEs66.shp"
                    LMEPolygon_path = os.path.abspath(LMEPolygon)
                    LME_sf = gpd.read_file(LMEPolygon_path)
                    LEM_gsd_new = LME_sf.to_crs(epsg=4326)

                    LME = pd.read_excel("LME values.xlsx")
                    LME.columns = LME.iloc[0]
                    LME = LME[1:].reset_index(drop=True)

                    AIS_long_lat = df_ais[['longitude', 'latitude']]
                    AIS_long_lat.columns = ['Longitude', 'Latitude']
                    points_cords = [Point(xy) for xy in zip(AIS_long_lat.Longitude, AIS_long_lat.Latitude)]
                    Route = gpd.GeoDataFrame(AIS_long_lat, geometry=points_cords, crs='EPSG:4326')

                    Route = gpd.sjoin(Route, LEM_gsd_new[['geometry', 'LME_NUMBER']], how="left", predicate='within')
                    Route['ID'] = Route['LME_NUMBER']
                    Route['Datetime'] = df_ais['DateTime']
                    result = pd.merge(Route, LME, how="left", on="ID")
                    result['months'] = result['Datetime'].apply(lambda x: x.strftime('%b'))

                    # Risk calculation
                    b = [0] * len(Route['geometry'])
                    Winter = ['Nov', 'Dec', 'Jan']
                    Spring = ['Feb', 'Mar', 'Apr']
                    Summer = ['May', 'Jun', 'Jul']
                    Autumn = ['Aug', 'Sep', 'Oct']

                    for i in range(len(result['geometry'])):
                        if result['months'][i] in Winter:
                            b[i] = result['Nov - Jan'][i]
                        elif result['months'][i] in Spring:
                            b[i] = result['Feb - Apr'][i]
                        elif result['months'][i] in Summer:
                            b[i] = result['May - Jul'][i]
                        else:
                            b[i] = result['Aug - Oct'][i]

                    df_ais['risk'] = b
                    st.subheader("📊 AIS with Risk Scores")
                    st.dataframe(df_ais)

                except Exception as e:
                    st.error(f"Geospatial or Excel error: {e}")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"Unexpected error during API call: {e}")

            # Time difference between points
            diff = [timedelta(0)] + list(df_ais['DateTime'].diff().iloc[1:])
            df_ais['Diff'] = diff
            
            # Distance in nautical miles (speed in knots * hours)
            df_ais['distance'] = df_ais['speed'] * 0.0002777778 * df_ais['Diff'].dt.total_seconds()
            
            # Sea Miles per Month (SMM)
            n = len(df_ais) - 1
            total_time_months = (df_ais['DateTime'].iloc[n] - df_ais['DateTime'].iloc[0]).total_seconds() / 2.628e+6
            smm = df_ais['distance'].sum() / total_time_months if total_time_months else 0
            
            # Total sea miles
            total_miles = df_ais['distance'].sum()
            
            # % Activity above 10 knots
            above_10 = df_ais[df_ais['speed'] > 10]
            perc_above_10 = (above_10['Diff'].sum() / df_ais['Diff'].sum()) * 100 if df_ais['Diff'].sum().total_seconds() > 0 else 0
            
            # % Activity above 3 knots
            above_3 = df_ais[df_ais['speed'] > 3]
            perc_above_3 = (above_3['Diff'].sum() / df_ais['Diff'].sum()) * 100 if df_ais['Diff'].sum().total_seconds() > 0 else 0
            
            # Summary table
            summary_df = pd.DataFrame({
                "Metric": ["Sea Miles per Month (SMM)", "Total Sea Miles", "% Time > 10 knots", "% Time > 3 knots"],
                "Value": [round(smm, 2), round(total_miles, 2), f"{perc_above_10:.2f}%", f"{perc_above_3:.2f}%"]
            })
            
            st.subheader("📈 Speed and Activity Summary")
            st.table(summary_df)

