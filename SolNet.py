# Do we need to reference PVLIB somehow when we use this? Or make our own version of the API connection?
import pvlib_helpers

class SolNet:
    """
    This class includes the model generation based on a destination provided by the user.
    """
    
    def sourceModel(latitude, longitude, features):
        
        print('Fetching Source Model data\n')
        
        source_data = pvlib_helpers.get_pvgis_hourly(latitude=latitude, 
                                                     longitude=longitude,
                                                     optimal_surface_tilt=True, 
                                                     optimalangles=True)
        
        print('Data gathered\n')
        
        return source_data