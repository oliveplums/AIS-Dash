import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import urllib3
import os
import sqlite3
from datetime import datetime
from typing import Sequence
from shapely.geometry import Point

# Disable HTTPS verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- API Functions ---

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
    
    # Filter, deduplicate, sort
    df1 = df_cleaned[df_cleaned["speed"] < 24].reset_index(drop=True)
    df = df1.drop_duplicates(subset='DateTime')
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    # Create GeoDataFrame for AIS route
    AIS_long_lat = df[['longitude', 'latitude']].copy()
    AIS_long_lat.columns = ['Longitude', 'Latitude']
    points_cords = [Point(xy) for xy in zip(AIS_long_lat.Longitude, AIS_long_lat.Latitude)]
    Route = gpd.GeoDataFrame(AIS_long_lat, geometry=points_cords, crs='EPSG:4326')
    
    # Ensure LME polygons are GeoDataFrame
    LEM_gsd_new = gpd.GeoDataFrame(LEM_gsd_new)
    
    # Assign LME OBJECTID to each AIS point
    a = [0] * len(Route)
    for i in range(len(Route)):
        for j in range(len(LEM_gsd_new)):
            if Route.geometry.iloc[i].within(LEM_gsd_new.geometry.iloc[j]):
                a[i] = LEM_gsd_new['OBJECTID'].iloc[j]
                break
    Route['ID'] = a
    
    # Assign datetime back
    Route['Datetime'] = df['DateTime']
    
    # Merge with LME Excel values
    result = pd.merge(Route, LME, how="left", on="ID")
    result['months'] = result['Datetime'].apply(lambda x: x.strftime('%b'))
    
    # Seasonal risk assignment
    b = [0] * len(result)
    Winter = ['Nov', 'Dec', 'Jan']
    Spring = ['Feb', 'Mar', 'Apr']
    Summer = ['May', 'Jun', 'Jul']
    Autumn = ['Aug', 'Sep', 'Oct']
    
    for i in range(len(result)):
        if result['months'].iloc[i] in Winter:
            b[i] = result['Nov - Jan'].iloc[i]
        elif result['months'].iloc[i] in Spring:
            b[i] = result['Feb - Apr'].iloc[i]
        elif result['months'].iloc[i] in Summer:
            b[i] = result['May - Jul'].iloc[i]
        else:
            b[i] = result['Aug - Oct'].iloc[i]
    
    # Assign back to final df
    df['risk'] = b
    
    # Optional: Show in UI
    st.subheader("🛟 AIS with LME Seasonal Risk")
    st.dataframe(df)

    return df_cleaned

# --- Streamlit UI ---

st.title("🚢 Voyage and AIS Dashboard with LME & Vessel Info")

username = st.secrets["username"]
password = st.secrets["password"]

imo_input = st.text_input("IMO number(s) (comma separated)", value="9770634")
start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
end_date = st.date_input("End Date", value=datetime(2024, 1, 10))
sixhourly = st.selectbox("6-Hourly data?", options=["true", "false"], index=0)

if st.button("Fetch Data"):
    with st.spinner("Fetching voyage history and AIS data..."):
        try:
            # Parse IMO input
            imo_list = list(map(int, imo_input.split(',')))

            # --- Fetch voyage + AIS ---
            voyage_data = get_voyage_history(username, password, imo_list, start_date, end_date)
            timestamp_changes = detect_mmsi_changes(voyage_data, end_date.isoformat())
            df_cleaned = fetch_and_combine_ais(username, password, timestamp_changes, start_date.isoformat(), end_date.isoformat(), sixhourly)

            if df_cleaned.empty:
                st.warning("No AIS position data found for the selected criteria.")
            else:
                st.success("AIS Data fetched successfully!")
                st.subheader("AIS Positions")
                st.dataframe(df_cleaned)
                st.map(df_cleaned.rename(columns={"latitude": "lat", "longitude": "lon"}))

            # --- Local SQLite vessel info ---
            with st.spinner("Querying local vessel info database..."):
                try:
                    imo_number = imo_list[0]
                    conn = sqlite3.connect(r"my_sqlite.db")
                    query = "SELECT * FROM vesselInfo WHERE LRIMOShipNo = ?"
                    dfVesselInfo = pd.read_sql(query, conn, params=(imo_number,))
                    conn.close()

                    if not dfVesselInfo.empty:
                        st.subheader("📄 Vessel Info (from local DB)")
                        st.dataframe(dfVesselInfo)
                    else:
                        st.info("No vessel info found in the local database.")
                except Exception as db_err:
                    st.error(f"SQLite error: {db_err}")

            # --- Load LME Polygon and Excel Info ---
            try:
                st.subheader("🌍 LME Polygon and Info")

                # Load shapefile
                LMEPolygon = "LMEPolygon1\LMEs66.shp"
                LME_sf = gpd.read_file(LMEPolygon).to_crs(epsg=4326)

                st.write("Loaded LME Polygons:")
                st.dataframe(LME_sf[['LME_NAME', 'geometry']].head())

                # Load Excel
                excel_path = r"C:\Users\palomboo\OneDrive - AkzoNobel\Template Automation Report\Intertrac Advance\LMEPolygon1\LME values.xlsx"
                LME = pd.read_excel(excel_path)
                LME.columns = LME.iloc[0]
                LME = LME[1:].reset_index(drop=True)

                st.write("Loaded LME Values from Excel:")
                st.dataframe(LME.head())

            except Exception as lme_err:
                st.error(f"Error loading LME data: {lme_err}")

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

