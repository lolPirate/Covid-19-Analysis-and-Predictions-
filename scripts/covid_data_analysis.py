import pandas as pd
import matplotlib.dates as mdates
import requests
from pprint import pprint
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('bmh')

COUNTRY = 'bangladesh'
STATUS = ['confirmed', 'recovered', 'deaths']
API = 'https://api.covid19api.com/dayone/country/{}/status/{}/live'
DATA_FOLDER_PATH = os.path.normpath(r'./data/')
PLOTS_FOLDER_PATH = os.path.normpath(r'./plots/')

data = {
    'cases': [],
    'date': [],
    'status': []
}

for stat in STATUS:
    api = API.format(COUNTRY, stat)
    r = requests.get(api)
    data_list = r.json()

    for day in data_list:
        data['cases'].append(day['Cases'])
        data['date'].append(day['Date'])
        data['status'].append(day['Status'])

data = pd.DataFrame.from_dict(data)
data = data.groupby(['date', 'status']).sum().reset_index()
data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%dT%H:%M:%SZ")

data.to_csv(os.path.join(DATA_FOLDER_PATH, f'{COUNTRY}.csv'), index=False)


def get_daily_cases(cases):
    '''
    Function to convert running total to discrete values
    '''
    cases_daily = []
    prev = 0
    for val in cases:
        actual = val - prev
        if actual < 0:
            actual = 0
        prev = val
        cases_daily.append(actual)
    return cases_daily


df_confirmed = data[data.status == 'confirmed'][['cases', 'date']]
cases_confirmed = df_confirmed.cases.values
cases_confirmed_daily = get_daily_cases(cases_confirmed)
dates_confirmed = df_confirmed.date.values

df_recovered = data[data.status == 'recovered'][['cases', 'date']]
cases_recovered = df_recovered.cases.values
cases_recovered_daily = get_daily_cases(cases_recovered)
dates_recovered = df_recovered.date.values

df_deaths = data[data.status == 'deaths'][['cases', 'date']]
cases_deaths = df_deaths.cases.values
cases_deaths_daily = get_daily_cases(cases_deaths)
dates_deaths = df_deaths.date.values

cases_active = cases_confirmed-cases_recovered-cases_deaths
#cases_active_daily = get_daily_cases(cases_active)
dates_active = dates_confirmed


years = mdates.YearLocator()
months = mdates.MonthLocator()
days = mdates.DayLocator()
years_fmt = mdates.DateFormatter('%Y')
month_fmt = mdates.DateFormatter('%M')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
ax1.plot(dates_confirmed, cases_confirmed, color='k', label='Confirmed')
ax1.plot(dates_recovered, cases_recovered, color='g', label='Recovered')
ax1.plot(dates_active, cases_active, '--', color='b', label='Active')
ax1.plot(dates_deaths, cases_deaths, color='r', label='Deaths')
ax1.xaxis.set_major_locator(months)
ax1.set_title(f'Cumulative')
ax1.legend(loc='best')

ax2.plot(dates_confirmed, cases_confirmed_daily,
         color='k', label='Reported', alpha=0.6)
ax2.plot(dates_recovered, cases_recovered_daily,
         color='g', label='Recovered', alpha=0.6)
#ax2.plot(dates_active, cases_active_daily, color='b', label='Active', marker='.', alpha=0.6)
ax2.plot(dates_deaths, cases_deaths_daily,
         color='r', label='Deaths', alpha=0.6)
ax2.xaxis.set_major_locator(months)
ax2.set_title(f'Daily Totals')
ax2.legend(loc='best')


date = dates_confirmed[-1].astype('datetime64[D]')
plt.suptitle(f'Covid-19 Data Analysis for {COUNTRY.upper()} as of {date}')
plt.savefig(os.path.join(PLOTS_FOLDER_PATH,
                         f'covid-19-analysis-{COUNTRY}-latest.jpg'), dpi=720, bbox_inches='tight')
plt.show()
