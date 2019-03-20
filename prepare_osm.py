# coding: utf-8


import csv
import pandas as pd
import numpy as np
import overpy


'''
OSM FEATURE COLLECTION
'''

# calculate bounding boxes

chi_crime_data = pd.read_csv('CHI_2015_2016_2017')
la_crime_data = pd.read_csv('LA_2015_2016_2017.csv')

chi_max_lat = chi_crime_data['Latitude'].max()
chi_min_lat = chi_crime_data['Latitude'].min()
chi_max_lon = chi_crime_data['Longitude'].max()
chi_min_lon = chi_crime_data['Longitude'].min()

chi_max_lat = chi_max_lat + .01
chi_min_lat = chi_min_lat - .01
chi_max_lon = chi_max_lon + .01
chi_min_lon = chi_min_lon - .01

la_max_lat = la_crime_data['Latitude'].max()
la_min_lat = la_crime_data['Latitude'].min()
la_max_lon = la_crime_data['Longitude'].max()
la_min_lon = la_crime_data['Longitude'].min()

la_max_lat = la_max_lat + .01
la_min_lat = la_min_lat - .01
la_max_lon = la_max_lon + .01
la_min_lon = la_min_lon - .01

print
print("Calculated Min and Max Coordinates for LA and Chicago")


api = overpy.Overpass()
def amenities_in_bbox(s,w,n,e):
    get_amenities = api.query("""<!--
        This has been generated by the overpass-turbo wizard.
        The original search was:
        “amenity=*”
        -->
        <osm-script output="json" timeout="25">
          <!-- gather results -->
          <union>
            <!-- query part for: “amenity=*” -->
            <query type="node">
              <has-kv k="amenity"/>
              <bbox-query s="{s}" w="{w}" n="{n}" e="{e}"/>
            </query>
            <query type="node">
              <has-kv k="shop"/>
              <bbox-query s="{s}" w="{w}" n="{n}" e="{e}"/>
            </query>
            <query type="way">
              <has-kv k="shop"/>
              <bbox-query s="{s}" w="{w}" n="{n}" e="{e}"/>
            </query>
            <query type="way">
              <has-kv k="amenity"/>
              <bbox-query s="{s}" w="{w}" n="{n}" e="{e}"/>
            </query>
            <query type="relation">
              <has-kv k="amenity"/>
              <bbox-query s="{s}" w="{w}" n="{n}" e="{e}"/>
            </query>
             <query type="relation">
              <has-kv k="shop"/>
              <bbox-query s="{s}" w="{w}" n="{n}" e="{e}"/>
            </query>
          </union>
          <!-- print results -->
          <print mode="body"/>
          <recurse type="down"/>
          <print mode="skeleton" order="quadtile"/>
        </osm-script>
        """.format(s=s, w=w, n=n, e=e))
    return get_amenities

chi_osm_indicators = amenities_in_bbox(chi_min_lat, chi_min_lon, chi_max_lat, chi_max_lon)
la_osm_indicators = amenities_in_bbox(la_min_lat, la_min_lon, la_max_lat, la_max_lon)

print("Retrieved OSM Data from Overpass")

# collect results and begin to unpack
# resolve missing true fixes a bug of missing data
chi_results = overpy.Result()
chi_nodes_list = chi_osm_indicators.get_nodes()
chi_ways_list = chi_osm_indicators.get_ways()
chi_relations_list = chi_osm_indicators.get_relations()

la_results = overpy.Result()
la_nodes_list = la_osm_indicators.get_nodes()
la_ways_list = la_osm_indicators.get_ways()
la_relations_list = la_osm_indicators.get_relations()

print("First step unpacking data done")

# make final lists 
chi_all_amenities = []
chi_lats = []
chi_lons = [] 

la_all_amenities = []
la_lats = []
la_lons = [] 

# process nodes and ways, the two major OSM features
for way in chi_osm_indicators.ways:
  amenity = way.tags.get("amenity", "n/a")
  if amenity != "n/a":
    amenity = amenity.encode('utf-8', "ignore")
    chi_all_amenities.append(amenity)
    temp_lats = []
    temp_lons = []
    for node in way.nodes:
      temp_lats.append(node.lat)
      temp_lons.append(node.lon)
    chi_lats.append(np.mean(temp_lats))
    chi_lons.append(np.mean(temp_lons))

for node in chi_osm_indicators.nodes:
  amenity = node.tags.get("amenity", "n/a")
  if amenity != "n/a":
    amenity = amenity.encode('utf-8', "ignore")
    chi_all_amenities.append(amenity)
    chi_lats.append(node.lat)
    chi_lons.append(node.lon)

for way in la_osm_indicators.ways:
  amenity = way.tags.get("amenity", "n/a")
  if amenity != "n/a":
    amenity = amenity.encode('utf-8', "ignore")
    la_all_amenities.append(amenity)
    temp_lats = []
    temp_lons = []
    for node in way.get_nodes(resolve_missing=True):
      temp_lats.append(node.lat)
      temp_lons.append(node.lon)
    la_lats.append(np.mean(temp_lats))
    la_lons.append(np.mean(temp_lons))

for node in la_osm_indicators.nodes:
  amenity = node.tags.get("amenity", "n/a")
  if amenity != "n/a":
    amenity = amenity.encode('utf-8', "ignore")
    la_all_amenities.append(amenity)
    la_lats.append(node.lat)
    la_lons.append(node.lon)

print("Second step unpacking done")
#print(len(la_osm_indicators.nodes))
#print(len(la_all_amenities))

chi_df = pd.DataFrame({ 'Indicator': pd.Series(chi_all_amenities), 'Latitude': pd.Series(chi_lats), 'Longitude': pd.Series(chi_lons)})
chi_df.to_csv('CHI_OSM.csv', index=False)

la_df = pd.DataFrame({ 'Indicator': pd.Series(la_all_amenities), 'Latitude': pd.Series(la_lats), 'Longitude': pd.Series(la_lons)})
la_df.to_csv('LA_OSM.csv', index=False)

print("Done Collecting OSM indicators for Chicago and Los Angeles, Saved as CSV")
