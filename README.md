# bike_routing_experiments
some experiments on bike routing

# Requirements
## Graphhopper
This requires a locally running instance of graphhopper.
I installed it according to https://github.com/graphhopper/graphhopper#installation

For my coordinates to work, you will need the Baden Württemberg map and start graphhopper with it.

    wget https://download.geofabrik.de/europe/germany/baden-wuerttemberg-latest.osm.pbf
    java -Ddw.graphhopper.datareader.file=baden-wuerttemberg-latest.osm.pbf -jar *.jar server config-example.yml

You can find my (very slightly modified) version of the config under `graphhopper_config/config-example.yml`. I only introduces cycling routing as an option.

## Python
Install the required python libraries with

    pip3 install -r requirements.txt

(tested under Ubuntu 20.04, python3.6)
