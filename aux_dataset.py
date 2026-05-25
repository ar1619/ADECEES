from netCDF4 import Dataset
import netCDF4 as nc
import xarray as xr
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import griddata
import time

RDS = '/scratch_hive/ar1619/RDS/earthdata/'
resampling = 'nearest'
year = 2017
half = 1

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_day_of_year(date_str):
    """
    Get the day of the year from a date string in YYYYMMDD format.

    Parameters:
    date_str (str): The date string in YYYYMMDD format.

    Returns:
    int: The day of the year.
    """
    date = datetime.strptime(date_str, '%Y%m%d')
    return date.timetuple().tm_yday

def get_area(coord, file, start_lat=90, end_lat=-90, start_lon=0, end_lon=360, win_size=256):
    """"""
    lon_dim = file.shape[1]
    lat_dim = file.shape[0]
    lat = coord[0]
    lon = coord[1]
    lat_h = (win_size/2)*180/(361*16)
    lon_h = (win_size/2)*360/(576*16)
    lats = np.linspace(start_lat, end_lat, lat_dim)
    lons = np.linspace(start_lon, end_lon, lon_dim)
    start_lat_idx = find_nearest(lats, lat-lat_h)
    end_lat_idx = find_nearest(lats, lat+lat_h)
    start_lon_idx = find_nearest(lons, lon-lon_h)
    end_lon_idx = find_nearest(lons, lon+lon_h)
    if start_lat_idx > end_lat_idx:
        return file[end_lat_idx:start_lat_idx, start_lon_idx:end_lon_idx]
    else:
        return file[start_lat_idx:end_lat_idx, start_lon_idx:end_lon_idx]

def resample_array(data, new_shape, resampling = 'nearest', masked_value=None):
    """
    Resample an array to a desired shape.

    Parameters:
    data (numpy.ndarray): The input array.
    new_shape (tuple): The desired shape of the output array.
    masked_value (float, optional): The value to use for masked elements.

    Returns:
    numpy.ndarray: The resampled array with the new shape.
    """
    mask = data.mask if np.ma.is_masked(data) else None
    if mask is not None:
        mask = resample_array(mask.astype(float), new_shape).astype(bool)
    data = np.asarray(data)
    old_shape = data.shape
    old_lat = np.linspace(0, 1, old_shape[0])
    old_lon = np.linspace(0, 1, old_shape[1])
    new_lat = np.linspace(0, 1, new_shape[0])
    new_lon = np.linspace(0, 1, new_shape[1])

    old_grid_lon, old_grid_lat = np.meshgrid(old_lon, old_lat)
    new_grid_lon, new_grid_lat = np.meshgrid(new_lon, new_lat)

    resampled_data = griddata((old_grid_lat.flatten(), old_grid_lon.flatten()), data.flatten(),
                              (new_grid_lat, new_grid_lon), method=resampling)

    if masked_value is not None:
        resampled_data[mask] = masked_value

    return resampled_data

def load_and_process_data(path, coordinates, date_year, date_day, variable='error'):
    start_lon = 0
    end_lon = 360
    if variable == 'temperature':
        file = nc.Dataset(path+[f for f in os.listdir(path) if "A{}{:03d}".format(date_year, date_day) in f][0],'r')['LST_Day_CMG']
        fill_value = file._FillValue
        # print(variable, fill_value)
    elif variable == 'vegetation':
        try:
            file = nc.Dataset(path+[f for f in os.listdir(path) if "A{}{:03d}".format(date_year, date_day) in f][0],'r')['CMG 0.05 Deg 16 days NDVI']
        except:
            day_quot = date_day // 16
            new_day = 16*day_quot + 1
            file = nc.Dataset(path+[f for f in os.listdir(path) if "A{}{:03d}".format(date_year, new_day) in f][0],'r')['CMG 0.05 Deg 16 days NDVI']
        fill_value = file._FillValue
        # print(variable, fill_value)
    else:
        fill_value = None
        start_lon = -180
        end_lon = 180
        file_global = xr.open_dataset(path)
        if variable == 'wind_u':
            file = file_global['u100'][date_day, :].values
            if np.isnan(file).any():
                file = file_global['u100'][date_day-1, :].values
        elif variable == 'wind_v':
            file = file_global['v100'][date_day, :].values
            if np.isnan(file).any():
                file = file_global['v100'][date_day-1, :].values
        elif variable == 'pressure':
            file = file_global['sp'][date_day, :].values
            if np.isnan(file).any():
                file = file_global['sp'][date_day-1, :].values
        elif variable == 'precipitation':
            file = file_global['tp'][date_day, :].values
            if np.isnan(file).any():
                file = file_global['tp'][date_day-1, :].values
    # file_norm = normalize_array(file[:], fill_value)
    # print("For {} on {}-{}: min: {}, max: {}".format(variable, date_year, date_day, np.min(file_norm), np.max(file_norm)))
    file_window = get_area(coordinates, file[:],  start_lon = start_lon, end_lon = end_lon,win_size=256)
    # file_window_norm = normalize_array(file_window, fill_value)
    # print("For {} on {}-{}: min: {}, max: {}".format(variable, date_year, date_day, np.min(file_window), np.max(file_window)))
    file_window_reshaped = resample_array(file_window, (256, 256))
    
    return file_window_reshaped

def add_additional_datasets(array, coordinates, data_paths, date_list):
    date_0 = date_list[0]
    date_year = date_0[:4]
    # temp_average = np.zeros((256,256))
    veg_average = np.zeros((256,256))
    wind_u_average = np.zeros((256,256))
    wind_v_average = np.zeros((256,256))
    press_average = np.zeros((256,256))
    precip_average = np.zeros((256,256))

    for date in date_list:
        date_day = get_day_of_year(date)
        if date_day < 180:
            folder = '{}_1'.format(date_year)
        elif date_day == 180:
            folder = '{}_2'.format(date_year)
            date_day = 1
        elif date_day == 181:
            folder = '{}_2'.format(date_year)
            date_day = 1
        else:
            folder = '{}_2'.format(date_year)
            date_day = date_day - 180
        # temperature Ayeardayoutof365
        # temp_average += load_and_process_data(data_paths[0], coordinates, date_year, date_day, variable='temperature')
        # vegetation Ayeardayoutof365
        veg_average += load_and_process_data(data_paths[1], coordinates, date_year, date_day, variable='vegetation')
        # wind_u year_half/100m_u_component_of_wind_0_daily-mean.nc
        path_wind_u = data_paths[2]+'{}/100m_u_component_of_wind_0_daily-mean.nc'.format(folder)
        wind_u_average += load_and_process_data(path_wind_u, coordinates, date_year, date_day, variable='wind_u')
        # wind_v year_half/100m_v_component_of_wind_0_daily-mean.nc
        path_wind_v = data_paths[2]+'{}/100m_v_component_of_wind_0_daily-mean.nc'.format(folder)
        wind_v_average += load_and_process_data(path_wind_v, coordinates, date_year, date_day, variable='wind_v')
        # pressure year_half/surface_pressure_stream-oper_daily-mean.nc
        path_press = data_paths[2]+'{}/surface_pressure_stream-oper_daily-mean.nc'.format(folder)
        press_average += load_and_process_data(path_press, coordinates, date_year, date_day, variable='pressure')
        # precipitation year_half/total_precipitation_0_daily-mean.nc
        path_precip = data_paths[2]+'{}/total_precipitation_0_daily-mean.nc'.format(folder)
        precip_average += load_and_process_data(path_precip, coordinates, date_year, date_day, variable='precipitation')
    # stack array with additional datasets
    array = np.stack((array, veg_average, wind_u_average, wind_v_average, press_average, precip_average), axis=0)
    return array

def main(coords, year=2017, half=1):
    win_size = 256
    folder = '{}_{}'.format(year, half)
    data_paths = [RDS+'temperature/', RDS+'vegetation/', RDS+'wind_press_precip/']
    root_grp = Dataset('auxiliary_2017.nc', 'w', format='NETCDF4')
    root_grp.description = 'Temperature, wind, pressure, precipitation and NDVI data for 2017'

    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('lats', 256)
    root_grp.createDimension('lons', 256)

    # variables
    time = root_grp.createVariable('time', 'f8', ('time',))
    lons = root_grp.createVariable('lons', 'f4', ('lons',))
    lats = root_grp.createVariable('lats', 'f4', ('lats',))
    wind_u = root_grp.createVariable('wind_u', 'f8', ('time', 'lats', 'lons',))
    wind_v = root_grp.createVariable('wind_v', 'f8', ('time', 'lats', 'lons',))
    press = root_grp.createVariable('pressure', 'f8', ('time', 'lats', 'lons',))
    precip = root_grp.createVariable('precipitation', 'f8', ('time', 'lats', 'lons',))
    temp = root_grp.createVariable('temperature', 'f8', ('time', 'lats', 'lons',))
    ndvi = root_grp.createVariable('vegetation', 'f8', ('time', 'lats', 'lons',))

    # data
    lat = coords[0]
    lon = coords[1]
    lat_h = (win_size/2)*180/(361*16)
    lon_h = (win_size/2)*360/(576*16)
    start_lat = lat-lat_h
    end_lat = lat+lat_h
    start_lon = lon-lon_h
    end_lon = lon+lon_h
    lats_values = np.linspace(start_lat, end_lat, 256)
    lons_values = np.linspace(start_lon, end_lon, 256)
    lats[:] = lats_values
    lons[:] = lons_values
    time[:] = np.arange(0, 10, 1)

    wind_u_file = xr.open_dataset(RDS+'wind_press_precip/{}_{}/100m_u_component_of_wind_0_daily-mean.nc'.format(year, half))
    wind_v_file = xr.open_dataset(RDS+'wind_press_precip/{}_{}/100m_v_component_of_wind_0_daily-mean.nc'.format(year, half))
    press_file = xr.open_dataset(RDS+'wind_press_precip/{}_{}/surface_pressure_stream-oper_daily-mean.nc'.format(year, half))
    precip_file = xr.open_dataset(RDS+'wind_press_precip/{}_{}/total_precipitation_0_daily-mean.nc'.format(year, half))
    temperature_file = xr.open_dataset(RDS+'temperature/temperature_{}.nc'.format(year))

    ndvi_dir = os.listdir(RDS+'vegetation')

    new_shape = (5776, 9216) #361*16, 576*16
    aux_variables_year = np.empty((0, 6, 5776, 9216))

    for i in range(10):
        start = time.time()
        date_day = i
        veg_average += load_and_process_data(data_paths[1], coordinates, year, date_day, variable='vegetation')
        # wind_u year_half/100m_u_component_of_wind_0_daily-mean.nc
        path_wind_u = data_paths[2]+'{}/100m_u_component_of_wind_0_daily-mean.nc'.format(folder)
        wind_u_average += load_and_process_data(path_wind_u, coordinates, year, date_day, variable='wind_u')
        # wind_v year_half/100m_v_component_of_wind_0_daily-mean.nc
        path_wind_v = data_paths[2]+'{}/100m_v_component_of_wind_0_daily-mean.nc'.format(folder)
        wind_v_average += load_and_process_data(path_wind_v, coordinates, year, date_day, variable='wind_v')
        # pressure year_half/surface_pressure_stream-oper_daily-mean.nc
        path_press = data_paths[2]+'{}/surface_pressure_stream-oper_daily-mean.nc'.format(folder)
        press_average += load_and_process_data(path_press, coordinates, year, date_day, variable='pressure')
        # precipitation year_half/total_precipitation_0_daily-mean.nc
        path_precip = data_paths[2]+'{}/total_precipitation_0_daily-mean.nc'.format(folder)
        precip_average += load_and_process_data(path_precip, coordinates, year, date_day, variable='precipitation')
        try:
            wind_u = wind_u_file['u100'][i, :, :]
            wind_v = wind_v_file['v100'][i, :, :]
            press = press_file['sp'][i, :, :]
            precip = precip_file['tp'][i, :, :]
            temp = temperature_file['t2m'][i, :, :]
        except:
            print(f"Error for day {i}")
        j = i + 1
        if j%16 == 1:
            try:
                ndvi_filename = [RDS+'vegetation/'+s for s in ndvi_dir if 'MOD13C1.A{}xxx'.format(year).replace('xxx', f"{j:03}") in s][0]
                ndvi = nc.Dataset(ndvi_filename, 'r')['CMG 0.05 Deg 16 days NDVI'][:]
                ndvi_time = time.time()
                ndvi_resampled = resample_array(ndvi, new_shape, -1e20)
                print(f"Ndvi resampling took {time.time()-ndvi_time} seconds")
            except:
                print(f"Error on NDVI for day {i}")
        time0 = time.time()
        wind_u_resampled = resample_array(wind_u, new_shape)
        time1 = time.time()
        wind_v_resampled = resample_array(wind_v, new_shape)
        time2 = time.time()
        press_resampled = resample_array(press, new_shape)
        time3 = time.time()
        precip_resampled = resample_array(precip, new_shape)
        time4 = time.time()
        temp_resampled = resample_array(temp, new_shape, 0)
        time5 = time.time()
        print(f"Wind u resampling took {time1-time0} seconds")
        print(f"Wind v resampling took {time2-time1} seconds")
        print(f"Pressure resampling took {time3-time2} seconds")
        print(f"Precipitation resampling took {time4-time3} seconds")
        print(f"Temperature resampling took {time5-time4} seconds")

        aux_variables = np.stack((wind_u_resampled, wind_v_resampled, press_resampled, precip_resampled, temp_resampled, ndvi_resampled), axis=0)
        aux_variables_year = np.concatenate((aux_variables_year, aux_variables[np.newaxis, :, :, :]), axis=0)
        print(f"Day {i} took {time.time()-start} seconds")
    return aux_variables_year


def make_nc_file(coords):
    root_grp = Dataset('auxiliary_2017.nc', 'w', format='NETCDF4')
    root_grp.description = 'Temperature, wind, pressure, precipitation and NDVI data for 2017'

    ndim = 256 # Size of the matrix ndim*ndim
    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('lats', ndim)
    root_grp.createDimension('lons', ndim)

    # variables
    time = root_grp.createVariable('time', 'f8', ('time',))
    x = root_grp.createVariable('lons', 'f4', ('lons',))
    y = root_grp.createVariable('lats', 'f4', ('lats',))
    field = root_grp.createVariable('field', 'f8', ('time', 'lons', 'lats',))

    # data
    lat = coords[0]
    lon = coords[1]
    x_range =  np.linspace(, 180, ndim)
    y_range =  np.linspace(, ydimension, ndim)
    x[:] = x_range
    y[:] = y_range
    for i in range(5):
        time[i] = i*50.0
        field[i,:,:] = np.random.uniform(size=(len(x_range), len(y_range)))


    root_grp.close()
print("File is being saved here: ", RDS+'aux_variables_{}_{}.npy'.format(year, half))
np.save(RDS+'aux_variables_{}_{}.npy'.format(year, half), main(year, half))