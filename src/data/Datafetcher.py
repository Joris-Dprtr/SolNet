import sys
import pickle
import datetime
import numpy as np
from ..util import formulas as fm
from ..util import pvgis_api as pvgis


class PvFetcher:

    def __init__(
            self,
            latitude,
            longitude,
            peakPower,
            tilt,
            azimuth,
            locations=5,
            start_date=2005,
            end_date=None,
            optimal_angles=None,
            km_radius=50,  # The radius around the actual location to find additional locations
            gaus_radius=0.5,  # The covariance for gaussian noise in km on the radius
            precision=40):

        """
        A class that gathers data from a location (and several neighbouring locations) on PVGIS, using the PVGIS API
        utility (see util).
        :param latitude: latitude of the location of interest
        :param longitude: longitude of the location of interest
        :param peak_power: the peak power of the installation
        :param tilt: the tilt of the solar panels
        :param azimuth: the direction of the solar panels
        :param start_date: the start date to gather data (minimum = 2005)
        :param locations: total locations for which we gather data. 5 locations means the base location + 4 others
        :param km_radius: the radius around the original location to find additional locations
        :param gaus_radius: a gaussian noise factor to include in the km_radius
        :param precision: the precision for when a location is located close to the sea
        """
        self.km_radius = km_radius
        self.gaus_radius = gaus_radius
        self.precision = precision

        self.pv = pvgis.PVgis(latitude, longitude, start_date, tilt, azimuth, peakPower, end=end_date,
                              optimalangles=optimal_angles)

        self.dataset = self._data_gathering(latitude, longitude, locations=locations)

    def _data_gathering(self, latitude, longitude, locations=5):
        """
        Gather data for a specific location and (if locations > 1) nearby locations
        :param latitude: latitude of the location of interest
        :param longitude: longitude of the location of interest
        :param locations: total locations for which we gather data. 5 locations means the base location + 4 others
        :return: list of length locations with PV + weather data for each of them
        """
        data = []

        additional_locations = fm.circle_pattern(locations - 1)

        # Base location

        print('Gathering data from base location...')

        try:
            data.append(self.pv.get_pvgis_hourly())

            if sum(data[0]['T2m']) == 0:
                raise ValueError("Location has no weather data, trying different location" + "...")

        except ValueError as ve:
            print(ve)
            sys.exit(1)

        except TimeoutError:
            print('Cannot connect to PVGIS')
            sys.exit(1)

        except Exception as e:
            print(e)
            sys.exit(1)

        # Additional locations

        lat_dif = fm.km_to_lat(self.km_radius)
        lat_dif_gaus = fm.km_to_lat(self.gaus_radius)

        # WORK IN PROGRESS: still have to finalise

        for i in range(locations - 1):

            print(f'Gathering data from additional location {i + 1}...')

            # The distance from the base location, transforming latitude and longitude to kilometers
            lat_additional = latitude + additional_locations['Sine'][i] * lat_dif

            # Longitude is based on the actual latitude and has to be calculated in the loop
            long_dif = fm.km_to_long(self.km_radius, lat_additional)
            long_additional = longitude + additional_locations['Cosine'][i] * long_dif

            # Gaussian randomisation longitude (has to be calculated in the loop) 
            long_dif_gaus = fm.km_to_long(self.gaus_radius, lat_additional)

            mean = [long_additional, lat_additional]
            cov = [[long_dif_gaus, 0], [0, lat_dif_gaus]]

            x, y = np.random.multivariate_normal(mean, cov, 1).T
            long_additional, lat_additional = x[0], y[0]

            # Check if location is on land 
            # If yes: append to the list

            long_arr = longitude + (long_additional - longitude) / self.precision * np.arange(self.precision)
            lat_arr = latitude + (lat_additional - latitude) / self.precision * np.arange(self.precision)

            long_list = long_arr.tolist()
            lat_list = lat_arr.tolist()

            for i in range(0, (self.precision - 1)):
                try:
                    long_new = long_list[-(i + 1)]
                    lat_new = lat_list[-(i + 1)]

                    # TO DO: add the extra locations

                    data.append(self.pv.get_pvgis_hourly())

                    if sum(data[-1]['T2m']) == 0:
                        del (data[-1])
                        raise ValueError("Location has no weather data, trying different location" + "...")
                    else:
                        break

                except ValueError as ve:
                    print(ve)

                except:
                    print('Location over sea, trying different location' + '...')

        pv_dataset_list = []

        for i in range(len(data)):
            data[i].index = data[i].index.tz_localize(None).floor('H')
            pv_dataset_list.append(data[i])

        return pv_dataset_list

    def save_data(self, file_name=None):
        """
        Save the data in a pickle file
        :param file_name: give a name to the file (optional, if None, a datetime string is given)
        :return: returns the file name
        """
        now = datetime.datetime.now()
        date_string = now.strftime("%y%m%d_%H%M")
        if file_name is None:
            file_name = f"dataset_{date_string}.pkl"
        with open('../data/' + file_name, "wb") as f:
            pickle.dump(self.dataset, f)

        return file_name
