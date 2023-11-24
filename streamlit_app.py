# import altair as alt
import time
import numpy as np
import pandas as pd
import ast
from datetime import datetime


import pickle
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from stravalib.client import Client # https://github.com/stravalib/stravalib


"""
# Strava Activity Cleaner
Streamlit app to fetch strava activities and clean gps from "standing still" datapoints.

"""

def save_to_pickle(obj, file_path='data/users.pickle'):
    """
    Save the given object to a pickle file.

    Parameters:
    - obj: The object to be saved.
    - file_path: The path to the pickle file. Default is 'data/users.pickle'.
    """
    Path(file_path).parent.mkdir(exist_ok=True, parents=True)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_from_pickle(file_path='data/users.pickle'):
    """
    Load an object from a pickle file.

    Parameters:
    - file_path: The path to the pickle file. Default is 'data/users.pickle'.

    Returns:
    - The loaded object, or None if the file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        return None



def convert_utc_to_string(utc_seconds):
    # Convert UTC seconds to a datetime object
    utc_datetime = datetime.utcfromtimestamp(utc_seconds)
    
    # Format the datetime object as a string
    formatted_time = utc_datetime.strftime("%d.%m.%Y %H:%M:%S")
    
    return formatted_time


min_speed = st.slider("Minmum speed in kph", min_value=0, max_value=15, value=3, step=1, format='%i km/h')
num_activities = st.slider("Number of activities to fetch", min_value=5, max_value=100, value=10, step=5) 

# st.write(f"Strava client ID is {st.secrets.strava.CLIENT_ID}")

MY_STRAVA_CLIENT_ID = st.secrets.strava.CLIENT_ID
MY_STRAVA_CLIENT_SECRET =  st.secrets.strava.CLIENT_SECRET
if hasattr(st.secrets, "strava.REDIRECT_URI_DEV"):
    REDIRECT_URI = st.secrets.strava.REDIRECT_URI_DEV
else:
    REDIRECT_URI = st.secrets.strava.REDIRECT_URI

SCOPE =   ast.literal_eval(st.secrets.strava.SCOPE)

# MY_STRAVA_CLIENT_ID
# MY_STRAVA_CLIENT_SECRET
REDIRECT_URI
# SCOPE


# load from pickle
client = load_from_pickle() if load_from_pickle() != None else Client()
if client == None:
    st.info('Login neassary. Please click auth strava button')
elif hasattr(client, "token_expires_at"):
    if time.time() < client.token_expires_at:
        st.success(f'Using already available login from pickle. expires at {convert_utc_to_string(client.token_expires_at)}')
    else:
        st.warning(f'Already available login expired at {convert_utc_to_string(client.token_expires_at)}. Please auth again')


url = client.authorization_url(client_id=MY_STRAVA_CLIENT_ID, redirect_uri=REDIRECT_URI, scope=SCOPE)

st.link_button("Authenticate with Strava", url)

query_parms = st.experimental_get_query_params()

auth_code = st.text_input("auth_code")
auth_code

access_token = False


# https://docs.streamlit.io/library/api-reference/utilities/st.experimental_get_query_params
if "code" in query_parms:
    st.text(query_parms["code"])
    auth_code = query_parms["code"]

if auth_code and not hasattr(client, "token_expires_at"):
    access_token = client.exchange_code_for_token(client_id=MY_STRAVA_CLIENT_ID, client_secret=MY_STRAVA_CLIENT_SECRET, code=auth_code)
    st.text(access_token)
    
    client.access_token = access_token['access_token']
    client.refresh_token = access_token['refresh_token']
    client.token_expires_at = access_token['expires_at']

    save_to_pickle(client)



if hasattr(client, "token_expires_at"):
    if time.time() > client.token_expires_at:
        refresh_response = client.refresh_access_token(
            client_id=MY_STRAVA_CLIENT_ID, client_secret=MY_STRAVA_CLIENT_SECRET, refresh_token=client.refresh_token
        )
        access_token = refresh_response["access_token"]
        refresh_token = refresh_response["refresh_token"]
        expires_at = refresh_response["expires_at"]

        save_to_pickle(client)

    else:        
        athlete = client.get_athlete()
        st.text("Athlete's name is {} {}, based in {}, {}"
            .format(athlete.firstname, athlete.lastname, athlete.city, athlete.country))
        
        
        activities = client.get_activities(limit=num_activities)

        # print(activities)
        # data = []
        # my_cols =[
        #   'name',
        #   'start_date_local',
        #   'type',
        #   'distance',
        #   'moving_time',
        #   'elapsed_time',
        #   'total_elevation_gain',
        #   'elev_high',
        #   'elev_low',
        #   'average_watts',
        #   'device_watts',
        #   'kilojoules',
        #   'average_speed',
        #   'max_speed',
        #   'average_heartrate',
        #   'max_heartrate',
        #   'start_latlng',
        #   'end_latlng',
        #   'kudos_count',
        #   ]
        data = {}
        for activity in activities:
            data.update({activity.id: activity.to_dict() })
         
        #data
        # Add id to the beginning of the columns, used when selecting a specific activity
        df = pd.DataFrame.from_records(data)
        df = df.transpose()

        #print(df)
        # Convert specific timedelta columns to seconds
        # time_columns = ['moving_time', 'elapsed_time']
        # Make all walks into hikes for consistency
        # df['type'] = df['type'].replace('Walk', 'Hike')
        # df['distance_km'] = df['distance']/1e3 
        # # Convert dates to datetime type
        # # df['start_date_local'] = pd.to_datetime(df['start_date_local'])
        # # # Create a day of the week and month of the year columns
        # df['day_of_week'] = df['start_date_local'].dt.day_name()
        # # df['month_of_year'] = df['start_date_local'].dt.month
        # # # Convert times to timedeltas
        # # df['moving_time'] = pd.to_timedelta(df['moving_time'])
        # # df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])
        # # # Convert timings to hours for plotting
        # df['elapsed_time_hr'] = df['elapsed_time'].astype(int)/3600e9
        # df['moving_time_hr'] = df['moving_time'].astype(int)/3600e9
        #st.dataframe(df)
        # components.html(df.to_html().replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" style="color: white"; "font-family: sans-serif">'), height=800, scrolling=True)


        # TODO make output nicer with date in german format + start time + name + id in the end 
        # ID == INDEX
        option = st.selectbox(
            'Select Activity',
            df.index, format_func=lambda x: f"{x} {(df.loc[df.index == x][['name', 'start_date_local']].values[0])} ")

        st.write('You selected:', option)

        client.get_activity(option)


        # Activities can have many streams, you can request n desired stream types
        types = [
            "time",
            "distance",
            "latlng",
            "altitude",
            "heartrate",
            "temp",
            "velocity_smooth",
            "moving",
            "watts",
            "cadence"
        ]

        resolutions = ['low', 'medium', 'high']
        res = st.selectbox('Select resolution', resolutions)

        # https://developers.strava.com/docs/reference/#api-Streams-getActivityStreams
        streams = client.get_activity_streams(option, types=types, resolution=res)

 
        #  Result is a dictionary object.  The dict's key are the stream type.
        for stream_type in streams.keys():
                st.markdown(f'## {stream_type}')
                st.markdown(f'containing {len(streams[stream_type].data)} datapoints')
                if stream_type == 'latlng':             
                        #print(streams[stream_type].data)
                        df = pd.DataFrame(streams[stream_type].data, columns=["lat", "lon"])
                        'scatter plot'
                        # TODO fix so it zooms in at the correct point
                        st.scatter_chart(df, y='lat', x='lon', )
                        ' map'
                        st.map(df, size=1)
                elif stream_type == 'velocity_smooth':
                    df = pd.DataFrame(streams[stream_type].data, columns=["velocity_smooth"])
                    # convert from m/s to kph
                    df["velocity_smooth"] = 3.6 * df["velocity_smooth"]
                    f'count of speeds below or equal {min_speed} kph is {len(df[(df["velocity_smooth"] <= min_speed)])}'
                    st.line_chart(df)
                else:
                    st.line_chart(streams[stream_type].data)

            #% TODO calc avg speed
            # TODO calc avg without "slow" datapoints
            # TODO generate new actviy and upload
            # DELETE old one probably needed?
                     
       
        # https://github.com/randyzwitch/streamlit-folium


# indices = np.linspace(0, 1, num_points)
# theta = 2 * np.pi * num_turns * indices
# radius = indices

# x = radius * np.cos(theta)
# y = radius * np.sin(theta)

# df = pd.DataFrame({
#     "x": x,
#     "y": y,
#     "idx": indices,
#     "rand": np.random.randn(num_points),
# })

# st.altair_chart(alt.Chart(df, height=700, width=700)
#     .mark_point(filled=True)
#     .encode(
#         x=alt.X("x", axis=None),
#         y=alt.Y("y", axis=None),
#         color=alt.Color("idx", legend=None, scale=alt.Scale()),
#         size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
#     ))
