# Geotracking
Stop Detection on Time Series Coordinate Data

This code calculates stoppages using time series data of latitude and langitude coordinates. It needs to be specialized for specific use cases.

## Preparation
Required input format:
User[String]\tSession[String]\tTimestamp[yyyy-mm-dd hh:mm:ss]\tlatitude[double]\tlongitude[double]

The code first sorts data by timestamp. We use density based clustering (DBSCAN) as that allows us to handle noise in the data.

## Likely Clusters

The algorithm for DBSCAN requires the specification of two parameters. The first one, epsilon, determines the neighborhood radius to consider for inclusion of points around a point, while the second one is the number of minimum points required around a point, within radius epsilon, for the identification of the point as a 'core point'.

For the example data whose origin is unknown we infer a range of values for both these parameters using some heuristics. To get an idea of the scales of the data, we calculate the distance corresponding to the difference between the max and min latitude, and similarly for the latitude. We then consider a range of values of radius that were fractional values of the overall scale.

For the minimum samples, we use the periodicity in the time series data. For the example the data is measured every minute or so.
That coupled with the low accuracy for the data gives us a rough lower bound on the min_samples range.

Iterating over both ranges gave us a bunch of likely possibilities for the "correct" cluster.

## Cluster Identification

We use three heuristics to limit the number of likely candidates. The first one is based on the cluster size. Specifically, we calculate the maximum and minimum distances of cluster members from their corresponding centroids. We then put a lower bound on the minimum distance to account for the uncertainty in the data, and finally pick clusters with the minimum maximum distance from the remaining candidates.

To break ties in the remaining clusters with the same maximum distances, we calculate the most common cluster size, and picking only those clusters that corresponded to that cluster size.

The final tie breaker is the number of minimum samples. We pick the cluster which has the maximum number of min_samples. The assumption being that the greater the number of points, greater is the reliability of the data.

## Other

We have accuracy data available which is not used explicitly.

To run the code, please change the location of the input file in the 'main' method of the script.
