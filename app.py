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
from math import radians, cos, sin, asin, sqrt
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

    df1 = df_cleaned[df_cleaned["speed"] < 24].reset_index(drop=True)
    df = df1.drop_duplicates(subset='DateTime')
    df = df.sort_values('DateTime').reset_index(drop=True)

    def haversine(lon1, lat1, lon2, lat2):
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 3440.065  # Radius of earth in nautical miles
        return c * r

    # Sort by DateTime
    df_cleaned = df_cleaned.sort_values('DateTime').reset_index(drop=True)
    
    # Calculate distances between points
    distances = [0]  # first point has no previous
    for i in range(1, len(df_cleaned)):
        lon1, lat1 = df_cleaned.loc[i-1, ['longitude', 'latitude']]
        lon2, lat2 = df_cleaned.loc[i, ['longitude', 'latitude']]
        dist = haversine(lon1, lat1, lon2, lat2)
        distances.append(dist)
    df_cleaned['dist_nm'] = distances

    # Extract month abbreviation for grouping
    df_cleaned['month'] = df_cleaned['DateTime'].dt.strftime('%b')

    # Group by month and calculate summary metrics
    summary = df_cleaned.groupby('month').apply(lambda x: pd.Series({
        'Sea Miles': x['dist_nm'].sum(),
        'Pct > 3 knots': (x['speed'] > 3).mean() * 100,
        'Most Common Speed': x['speed'].mode().iloc[0] if not x['speed'].mode().empty else np.nan
    })).reset_index()

    # Add total sea miles
    total_sea_miles = df_cleaned['dist_nm'].sum()

    st.subheader("📊 Monthly AIS Summary")
    st.dataframe(summary)

    st.write(f"**Total Sea Miles Traveled:** {total_sea_miles:.2f} nm")

    return df

# ---- Stationary periods detection function (simplified) ----
def detect_stationary_periods(df, speed_threshold=0.3, min_duration_hours=12):
    df = df.sort_values('DateTime').reset_index(drop=True)
    df['stationary'] = df['speed'] <= speed_threshold

    stationary_periods = []
    start_idx = None

    for i, stationary in enumerate(df['stationary']):
        if stationary and start_idx is None:
            start_idx = i
        elif not stationary and start_idx is not None:
            duration = (df.loc[i-1, 'DateTime'] - df.loc[start_idx, 'DateTime']).total_seconds() / 3600
            if duration >= min_duration_hours:
                stationary_periods.append((df.loc[start_idx, 'DateTime'], df.loc[i-1, 'DateTime'], duration))
            start_idx = None

    # Catch if last period is stationary and still ongoing
    if start_idx is not None:
        duration = (df.loc[len(df)-1, 'DateTime'] - df.loc[start_idx, 'DateTime']).total_seconds() / 3600
        if duration >= min_duration_hours:
            stationary_periods.append((df.loc[start_idx, 'DateTime'], df.loc[len(df)-1, 'DateTime'], duration))

    return pd.DataFrame(stationary_periods, columns=['Start', 'End', 'DurationHours'])

# ---- Streamlit UI ----

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

            ais_df = fetch_and_combine_ais(username, password, timestamp_changes, start_date.isoformat(), end_date.isoformat(), sixhourly)
            
            if ais_df.empty:
                st.warning("No AIS position data found for the selected criteria.")
            else:
                st.success("AIS Data fetched successfully!")
                st.subheader("AIS Positions")
                st.dataframe(ais_df)
                st.map(ais_df.rename(columns={"latitude": "lat", "longitude": "lon"}))

                # Detect stationary periods
                stationary_df = detect_stationary_periods(ais_df)
                st.subheader("Stationary Periods (>12 hours, speed ≤ 0.3 knots)")
                st.dataframe(stationary_df)

                # Plot speed histogram
                fig_speed = px.histogram(ais_df, x='speed', nbins=50, title='Speed Distribution')
                st.plotly_chart(fig_speed)

                # Plot stationary periods duration histogram
                fig_stationary = px.histogram(stationary_df, x='DurationHours', nbins=30, title='Stationary Period Durations (hours)')
                st.plotly_chart(fig_stationary)

                # Optional: save stationary periods to Excel
                stationary_df.to_excel("stationary_periods.xlsx", index=False)
                st.markdown("Stationary periods exported as `stationary_periods.xlsx`.")

            # Query local vessel info (SQLite)
            try:
                imo_number = imo_list[0]
                conn = sqlite3.connect("my_sqlite.db")
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

            # Load LME polygons and risk data (optional, add your paths here)
            try:
                LMEPolygon = "LMEPolygon1/LMEs66.shp"  # fix path if needed
                LME_sf = gpd.read_file(LMEPolygon).to_crs(epsg=4326)
                st.subheader("🌍 LME Polygons loaded")
                st.dataframe(LME_sf[['LME_NAME']].head())

                excel_path = r"C:\Users\palomboo\OneDrive - AkzoNobel\Template Automation Report\Intertrac Advance\LMEPolygon1\LME values.xlsx"
                LME = pd.read_excel(excel_path)
                LME.columns = LME.iloc[0]
                LME = LME[1:].reset_index(drop=True)
                st.write("LME risk values loaded")
                st.dataframe(LME.head())

                # Here you can integrate the LME risk assignment to ais_df, similar to your previous code

            except Exception as lme_err:
                st.error(f"Error loading LME data: {lme_err}")

        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

