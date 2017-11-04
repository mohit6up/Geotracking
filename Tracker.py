import datetime, time
import numpy as np
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from pandas import DataFrame
from sklearn.cluster import DBSCAN

class Tracker:
"""Required input format of file xxx.txt is like the following:
User[String]\tSession[String]\tTimestamp[yyyy-mm-dd hh:mm:ss]\tlatitude[double]\tlongitude[double]
"""

    KMS_IN_RADIANS = 6371.0088
    INSTALL_ID = 'xxx'

    def __init__(self, file_path):
        df = DataFrame.from_csv(file_path, sep="\t")
        df['timestamp'] = df['time'].apply(lambda value: \
            time.mktime(datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timetuple()))
        self.df = df.sort_values('timestamp')
        self.coordinates = self.df.as_matrix(columns=['lat', 'lng'])

    def all_clusters(self):
        """
        Identify like clusters by iterating over a range of epsilon and min_samples values. The epsilon
        values are guided by the km range of the data provided. The min_samples considers the fact that the
        data is not very accurate. We also notice that the samples are taken every minute or so for the most 
        part.
        """
        self.dbs = Counter()
        for points in range(10, 25):
            for radius in range(1, 10):
                epsilon = (0.005 * radius)/self.KMS_IN_RADIANS
                self.dbs[(points, 0.005 * radius)] = DBSCAN(eps=epsilon, min_samples=points, \
                    metric='haversine').fit_predict(np.radians(self.coordinates))

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def distance_bounds(self, clusters, dframe, return_centers = False):
        """
        Identify max and minimum distance to centroids. Used to identify likely clusters.
        """
        latitudes = Counter()
        longitudes = Counter()
        counts = Counter()
        dframe['cluster'] = clusters
        for index, row in dframe.iterrows():
            latitudes[row['cluster']] += row['lat']
            longitudes[row['cluster']] += row['lng']
            counts[row['cluster']] += 1
        centers = Counter()
        for i in range(0, len(set(clusters)) - 1):
            centers[i] = (latitudes[i]/counts[i], longitudes[i]/counts[i])
        max_distances = Counter()
        for index, row in dframe.iterrows():
            if row['cluster'] != -1:
                distance_from_center = self.haversine(row['lng'], row['lat'], centers[row['cluster']][1], \
                    centers[row['cluster']][0])
                if distance_from_center > max_distances[row['cluster']]:
                    max_distances[row['cluster']] = distance_from_center
        if (return_centers):
            return (min(max_distances.values()), max(max_distances.values()), centers)
        else:
            return (min(max_distances.values()), max(max_distances.values()))

    def extremum_distances(self):
        self.all_clusters()
        maxes = Counter()
        for j in self.dbs.keys():
            maxes[j] = (self.distance_bounds(self.dbs[j], self.df), len(set(self.dbs[j])))
        return maxes

    def majority_cluster(self, arr):
        cnt = Counter()
        for i in arr:
            cnt[i] += 1
        return cnt.most_common(1)

    def smooth(self, k, arr):
        dc = arr
        for index, val in enumerate(dc):
            if (index + k) >= len(dc):
                continue
            if (dc[index] != dc[index + k]):
                continue
            elif self.majority_cluster(dc[index:index +k])[0][0] != dc[index]:
                continue
            else:
                for j in range(1, k - 1):
                    if dc[index + j] != dc[index]:
                        dc[index + j] = dc[index]
        return dc

    def most_likely_cluster(self):
        """
        We whittle down candidates using the following heuristics:
        1. First using feasible bounds on the min and maximum distance from centroid
        2. Second by using the most common cluster size
        3. And finally by using the maximum number of min_samples (for dbscan)
        """
        maxes = self.extremum_distances()
        viable = { k: maxes[k] for k in maxes.keys() if maxes[k][0][1] <= 0.03 and \
            maxes[k][0][0] >= 0.0001 and maxes[k][1] > 2 }
        most_common_max_distance = Counter([viable[x][0][1] for x in viable.keys()]).most_common(1)[0][0]

        remaining = { x: viable[x] for x in viable.keys() if viable[x][0][1] == most_common_max_distance }
        most_common_size = Counter([remaining[x][1] for x in remaining.keys()]).most_common(1)[0][0]

        remaining = { k: remaining[k] for k in remaining.keys() if remaining[k][1] == most_common_size }
        max_reliable = max([k[0] for k in remaining.keys()])
        final = self.dbs[[k for k in remaining.keys() if k[0] == max_reliable][0]]
        return final

    def prepare_output(self):
        df = self.df
        j = self.most_likely_cluster()
        # smooth the set of cluster assignments
        for i in range(2, 4):
            j = self.smooth(i, j)
        # identify indices corresponding to the start and end of clusters
        values = []
        previous = -2
        for i, val in enumerate(j):
            if val != previous:
                values.append((i, val))
                previous = val
        values.extend([((k[0] - 1, -2)) for k in values[1:]])
        indices = sorted([k[0] for k in values])
        final_pairs = [(i, j[i]) for i in indices if j[i] != -1]
        result = []
        # For final results use coordinates closest to centroid
        a, b, centers = self.distance_bounds(j, self.df, True)
        for j in range(len(final_pairs)/2):
            start_index = final_pairs[2 * j][0]
            end_index = final_pairs[2 * j + 1][0]
            from_center = [ self.haversine(df.iloc[i]['lng'], df.iloc[i]['lat'], centers[df.iloc[i]['cluster']][1], \
                centers[df.iloc[i]['cluster']][0]) for i in range(start_index, end_index)]
            index = from_center.index(min(from_center))
            result.append((self.INSTALL_ID, df.iloc[start_index]['time'], df.iloc[end_index]['time'], \
                df.iloc[start_index + index]['lat'], df.iloc[start_index + index]['lng']))
        return result

def main():
    from pandas import DataFrame
    tracker = Tracker('xxx.txt')
    df = DataFrame((tracker.prepare_output()))
    df.to_csv('stoppages.csv', index=False, header=False)

if __name__ =='__main__':
    main()
