import pandas as pd
import requests as req
from bs4 import BeautifulSoup  as bs
import re

df=req.get("https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")

# Parsing the HTML
soup = bs(df.content, 'html.parser')

s = soup.find('div', class_='entry-content')
content = soup.find_all('a')

taxi_urls=[]
zone_table_url=[]

for url in content:
    href=url.get("href","")
    href=href.strip()

    if 'yellow' in href or 'green' in href:
        if '2024-09' in href or '2024-11' in href or '2024-10' in href :
            #print(href)
            taxi_urls.append(href)
    if 'zone' in href and href[-3:]=='csv':
        zone_table_url.append((href))

print((taxi_urls),(zone_table_url))