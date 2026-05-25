import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from datetime import datetime, timedelta
import time
from multiprocessing import cpu_count, Pool

def split_data_loader(list_of_days, n_exclude=5, shuffle=True):
    """
    Splits the dataset and puts n_exclude samples in a separate list.
    """
    n_samples = len(list_of_days)
    # n_item = n_samples // Batch_Size
    n_train = n_samples - n_exclude

    indices = list(range(n_samples))
    if shuffle:
        np.random.shuffle(indices)


    if n_exclude >= n_samples:
        raise ValueError("Can't split the dataset: n_exclude must be less than the number of items in the dataset.")
    
    train_indices = indices[:n_train]
    exclude_indices = indices[n_train:]

    return [list_of_days[i] for i in train_indices], [list_of_days[i] for i in exclude_indices]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def array_centered_on_point(original_array, point, window_size=16):
    idx_lat = find_nearest(original_array['lat'], point[0])
    idx_lon = find_nearest(original_array['lon'], point[1])
    lat_start = original_array['lat'][idx_lat - window_size]
    lat_end = original_array['lat'][idx_lat + window_size - 1]
    lon_start = original_array['lon'][idx_lon - window_size]
    lon_end = original_array['lon'][idx_lon + window_size - 1]

    return original_array['XCO2'].sel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end)).values

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

def load_and_process_data(data_paths, coordinates, date_year, date_day, folder, variable='error'):
    start_lon = 0
    end_lon = 360
    if variable == 'vegetation':
        path = data_paths[1]
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
        if variable == 'temperature':
            path = data_paths[0]+'temperature_{}.nc'.format(date_year)
            file_global = xr.open_dataset(path)
            file = file_global['t2m'][date_day, :].values
        if variable == 'wind_u':
            path = data_paths[2]+'{}/100m_u_component_of_wind_0_daily-mean.nc'.format(folder)
            file_global = xr.open_dataset(path)
            file = file_global['u100'][date_day, :].values
            if np.isnan(file).any():
                file = file_global['u100'][date_day-1, :].values
        elif variable == 'wind_v':
            path = data_paths[2]+'{}/100m_v_component_of_wind_0_daily-mean.nc'.format(folder)
            file_global = xr.open_dataset(path)
            file = file_global['v100'][date_day, :].values
            if np.isnan(file).any():
                file = file_global['v100'][date_day-1, :].values
        elif variable == 'pressure':
            path = data_paths[2]+'{}/surface_pressure_stream-oper_daily-mean.nc'.format(folder)
            file_global = xr.open_dataset(path)
            file = file_global['sp'][date_day, :].values
            if np.isnan(file).any():
                file = file_global['sp'][date_day-1, :].values
        elif variable == 'precipitation':
            path = data_paths[2]+'{}/total_precipitation_0_daily-mean.nc'.format(folder)
            file_global = xr.open_dataset(path)
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

def process_date(results, date, var, data_paths, coordinates, date_year):
    date_day = get_day_of_year(date)
    if date_day < 180:
        folder = '{}_1'.format(date_year)
    elif date_day == 180 or date_day == 181:
        folder = '{}_2'.format(date_year)
        date_day = 1
    else:
        folder = '{}_2'.format(date_year)
        date_day = date_day - 180

    results[var] += load_and_process_data(data_paths, coordinates, date_year, date_day, folder, variable=var)

    return results

def add_additional_datasets(array, coordinates, data_paths, date_list):
    pool_size = Pool(64)
    pool = Pool(processes=64)
    # print("Pool_size: ", pool_size, "Number of processes: ", cpu_count())
    date_0 = date_list[0]
    date_year = date_0[:4]
    temp_average = np.zeros((256,256))
    veg_average = np.zeros((256,256))
    wind_u_average = np.zeros((256,256))
    wind_v_average = np.zeros((256,256))
    press_average = np.zeros((256,256))
    precip_average = np.zeros((256,256))

    variables = ['temperature', 'vegetation', 'wind_u', 'wind_v', 'pressure', 'precipitation']
    results = {'temperature': np.zeros((256,256)), 'vegetation': np.zeros((256,256)), 'wind_u': np.zeros((256,256)), 'wind_v': np.zeros((256,256)), 'pressure': np.zeros((256,256)), 'precipitation': np.zeros((256,256))}

    # results = Parallel(n_jobs=-1)(delayed(process_date)(date, data_paths, coordinates, date_year) for date in date_list)
    results_week = pool_size.starmap(process_date, [(results, date, var, data_paths, coordinates, date_year) for date in date_list for var in variables])
    print("Final results length: ", len(results_week))
    # print("Result day 1 shape: ", results_week[0].shape)
    # print("Result vegetation: ", results_week['blop'])


    for result in results_week:
        temp_average += result['temperature']
        veg_average += result['vegetation']
        wind_u_average += result['wind_u']
        wind_v_average += result['wind_v']
        press_average += result['pressure']
        precip_average += result['precipitation']
    # Put back veg_average when needed
    array = np.stack((array, temp_average, veg_average, wind_u_average, wind_v_average, press_average, precip_average), axis=0)
    return array

def main():
    pass


if __name__ == '__main__':
    main()