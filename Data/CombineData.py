"""
This class is designed to compile all the data into one Microsoft Access database.
The data from 2020/01/22 to 2020/03/24 are all from John Hopkins.
From 03/25 on it is data scraped from worlometer since they are the only ones proving recovered data
"""

import pandas as pd
from Communication import Database
database = Database("C:\\Users\\Chris\\Documents\\PycharmProjects\\COVID19\\Data\\CombinedData.accdb")

def insert_confirmed():
    data = pd.read_csv("csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_confirmed_global.csv")
    print(data)
    data.columns = data.columns.str.replace('/', '')

    for line in range(len(data)):
        province = data.iloc[line, 0]
        if pd.isna(province):
            province = ""
        country = data.iloc[line, 1]

        province = province.replace("'", "")
        country = country.replace("'", "")

        infected0324 = data["32420"][line]
        infected0325 = data["32520"][line]
        query = "INSERT INTO IRTbl(Province, Country, Day, Infected)\n" \
                "VALUES ('{}', '{}', #2020/03/24#, {});".format(province, country, infected0324)
        print(query)
        database.update_insert(query)
        query = "INSERT INTO IRTbl(Province, Country, Day, Infected)\n" \
                "VALUES ('{}', '{}', #2020/03/25#, {});".format(province, country, infected0325)
        database.update_insert(query)


def append_recovered():
    data = pd.read_csv("csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_recovered_global.csv")
    print(data)
    data.columns = data.columns.str.replace('/', '')

    for line in range(len(data)):
        province = data.iloc[line, 0]
        if pd.isna(province):
            province = ""
        country = data.iloc[line, 1]

        province = province.replace("'", "")
        country = country.replace("'", "")

        recovered0324 = data["3242020"][line]
        recovered0325 = recovered0324  # Since we don't have this data just use the previous day's data
        query = "UPDATE IRTbl SET IRTbl.Recovered = {}\n" \
                "WHERE (((IRTbl.[Province])='{}') AND ((IRTbl.[Country])='{}') " \
                "AND ((IRTbl.[Day])=#2020/03/24#));".format(recovered0324, province, country)
        print(query)
        database.update_insert(query)
        query = "UPDATE IRTbl SET IRTbl.Recovered = {}\n" \
                "WHERE (((IRTbl.[Province])='{}') AND ((IRTbl.[Country])='{}') " \
                "AND ((IRTbl.[Day])=#2020/03/25#));".format(recovered0325, province, country)
        database.update_insert(query)


def append_dead():
    data = pd.read_csv("csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_deaths_global.csv")
    print(data)
    data.columns = data.columns.str.replace('/', '')

    for line in range(len(data)):
        province = data.iloc[line, 0]
        if pd.isna(province):
            province = ""
        country = data.iloc[line, 1]

        province = province.replace("'", "")
        country = country.replace("'", "")

        dead0324 = data["32420"][line]
        dead0325 = data["32520"][line]
        query = "UPDATE IRTbl SET IRTbl.Dead = {}\n" \
                "WHERE (((IRTbl.[Province])='{}') AND ((IRTbl.[Country])='{}') " \
                "AND ((IRTbl.[Day])=#2020/03/24#));".format(dead0324, province, country)
        print(query)
        database.update_insert(query)
        query = "UPDATE IRTbl SET IRTbl.Dead = {}\n" \
                "WHERE (((IRTbl.[Province])='{}') AND ((IRTbl.[Country])='{}') " \
                "AND ((IRTbl.[Day])=#2020/03/25#));".format(dead0325, province, country)
        database.update_insert(query)


def insert_worldometer(day):
    data = pd.read_csv("Worldometer\\"+str(day)+"March.csv")
    print(data)
    data.columns = data.columns.str.replace('/', '')

    for line in range(len(data)):
        province = ""
        country = data.iloc[line, 0]

        province = province.replace("'", "")
        country = country.replace("'", "")
        country = country.replace("USA", "US")

        infected = data.iloc[line, 1]
        dead = data.iloc[line, 3]
        recovered = data.iloc[line, 5]

        if pd.isna(infected):
            infected = 0
        if pd.isna(dead):
            dead = 0
        if pd.isna(recovered):
            recovered = 0

        query = "INSERT INTO IRTbl(Province, Country, Day, Infected, Recovered, Dead)\n" \
                "VALUES ('{}', '{}', #2020/03/{}#, {}, {}, {});".format(province, country, day, infected, recovered, dead)
        print(query)
        database.update_insert(query)


insert_worldometer(27)
