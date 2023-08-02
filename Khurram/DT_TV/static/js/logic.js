
let url = '/static/station.json';
let torontoCoords = [43.79, -79.47]; 
let mapZoomLevel = 8;


// Create the createMap function.
function createMap (bstations) {

  // Create the tile layer that will be the background of our map.
  let street = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  });


  // Create the map object with options.
  let myMap = L.map("map-id", {
    center: torontoCoords,
    zoom: mapZoomLevel,
    layers: [street, bstations]
  })

  // Create a layer control, and pass it baseMaps and overlayMaps. Add the layer control to the map.
  L.control.layers({
    collapsed: false
  }).addTo(myMap)
}

// Create the createMarkers function.
function createMarkers(response) {

  // Pull the "stations" property from response.data.
  let stations = response.stations

  // Initialize an array to hold the bike markers.
  let bMarkers = []
  // console.log(stations)

  // Loop through the stations array.
  for (let i = 0; i < stations.length; i++) {
    let station = stations[i]
    let Coords = [station.lat, station.lon]

    // For each station, create a marker, and bind a popup with the station's name.
    let bMarker = L.marker(Coords).bindPopup("<h3>" + station.name + "<h3>")
    // let bMarker = L.marker(Coords)
    
    // Add the marker to the bMarkers array.
    // console.log(Coords)
    bMarkers.push(bMarker)
  }
  // Create a layer group that's made from the bmarkers array, and pass it to the createMap function.
  createMap(L.layerGroup(bMarkers))
}


// Perform an API call to the Citi Bike API to get the station information. Call createMarkers when it completes.
d3.json(url).then(createMarkers);



