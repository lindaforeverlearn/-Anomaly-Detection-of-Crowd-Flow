import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import matplotlib.pyplot as plt
import os
from interval import Interval
# import missingno as msno
from scipy import interpolate

location = 'Taipei101' 


# -------------- test -----------
def test_fakevsreal():
    for epoch in range(30):  # 30
        print("Epoch is　：", epoch)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("Fake vs Real" + str(epoch))
        fake = np.load(f'./chungshan/GAN/npy/fake/{str(epoch)}.npy')
        fake = fake.reshape(3, 3)
        sns.heatmap(ax=ax[0], data=fake, linewidths=0.5, vmin=500, vmax=10000)
        ax[0].set_title('Fake')
        real = np.load(f'./chungshan/GAN/npy/real/{str(epoch)}.npy')
        real = real.reshape(3, 3)
        sns.heatmap(ax=ax[1], data=real, linewidths=0.5, vmin=500, vmax=10000)
        ax[1].set_title('Real')
        plt.show()

def plot_choropleth(data):
    year = '2018'
    import folium

    fmap = folium.Map(location=[25.050, 121.520], zoom_start=15)
    folium.Choropleth(
        data=data,
        name='choropleth',
        columns=['Grid500', 'mark'],
        threshold_scale=[-1500, -1000, -500, 0, 500, 1000, 1500],
        key_on='feature.id',
        fill_color='YlGnBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=year + '25.050, 121.520',
        reset=True).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    item_txt = """<br><i style="color:{col}"></i>"""
    legend_html = """
         <div style="
         position: fixed;
         bottom: 200px; left: 790px; width: 500px; height: 45px;
         z-index:9999;

         background-color:white;
         opacity: .65;

         font-size:28px;
         font-weight: bold;

         ">
         &nbsp; {title}
          </div> """.format(title='location=[25.050, 121.520]', itm_txt=item_txt)
    fmap.get_root().html.add_child(folium.Element(legend_html))
    file_path = 'C:/Users/Linda/Desktop/Master/model/Code/People_GAN_pycharm.html'
    fmap.save(file_path)

def sigma(data, n):
    data = data.POPULATION.tolist()

    ymean = np.mean(data)  
    ystd = np.std(data)  
    up = ymean + n * ystd
    outlier = []  
    index_list = []

    for i in range(0, len(data)):
        if data[i] > up:
            outlier.append(data[i])
            index_list.append(i)
        else:
            continue
    return outlier, index_list, up


def sigma_create_data(n, lon, lat):
    if not os.path.exists(f'./{location}/{n}sig'):
        os.makedirs(f'./{location}/{n}sig')
    train_threshold = pd.DataFrame()
    delta = 0.005
    right = str('%.03f' % (lon + delta))
    left = str('%.03f' % (lon - delta))
    top = str('%.03f' % (lat + delta))
    down = str('%.03f' % (lat - delta))
    lon = str('%.03f' % lon)
    lat = str('%.03f' % lat)

    left_top, top, right_top = f'{left}_{top}', f'{lon}_{top}', f'{right}_{top}'
    left_center, center, right_center = f'{left}_{lat}', f'{lon}_{lat}', f'{right}_{lat}'
    left_down, down, right_down = f'{left}_{down}', f'{lon}_{down}', f'{right}_{down}'
    location_list = ['left_top', 'top', 'right_top', 'left', 'center', 'right', 'left_down', 'down', 'right_down']
    loc_grid_list = [left_top, top, right_top, left_center, center, right_center, left_down, down, right_down]

    grid = 1
    z = np.zeros((8760, 3, 3))
    index_1 = 0
    index_2 = 0
    for i, loc in enumerate(loc_grid_list):
        gr = pd.read_csv(f'./grid_data_train/{str(loc)}.csv')
        outlier, index_list, up = sigma(gr, n)
        train_threshold.at[i, 'location'] = location_list[i]
        train_threshold.at[i, 'threshold'] = up

        for index in index_list:
            gr.at[index, 'POPULATION'] = np.nan

        gr['POPULATION'] = gr['POPULATION'].interpolate(method='linear')
        gr['POPULATION'] = gr['POPULATION'].fillna(gr['POPULATION'].mean())
        gr.to_csv(f'./{location}/{n}sig/{str(grid)}_{n}sig.csv', index=False, encoding='utf-8-sig')

        for hr in range(8760):
            z[hr][index_1][index_2] = gr.POPULATION[hr]
        grid += 1
        index_2 += 1
        if index_2 == 3:
            index_2 = 0
            index_1 += 1

    train_threshold.to_csv(f'./{location}/{n}sig/threshold_{str(center)}.csv', index=False, encoding='utf-8-sig')
    np.save(f'./{location}/save_z_{n}sig.npy', z)



def save_z_last():
    gr1 = gr1_csv.POPULATION
    gr2 = gr2_csv.POPULATION
    gr3 = gr3_csv.POPULATION
    gr4 = gr4_csv.POPULATION
    gr5 = gr5_csv.POPULATION
    gr6 = gr6_csv.POPULATION
    gr7 = gr7_csv.POPULATION
    gr8 = gr8_csv.POPULATION
    gr9 = gr9_csv.POPULATION
    z = np.zeros((8760, 3, 3))
    for hr in range(8760):
        z[hr][0][0] = gr1[hr]
        z[hr][0][1] = gr2[hr]
        z[hr][0][2] = gr3[hr]
        z[hr][1][0] = gr4[hr]
        z[hr][1][1] = gr5[hr]
        z[hr][1][2] = gr6[hr]
        z[hr][2][0] = gr7[hr]
        z[hr][2][1] = gr8[hr]
        z[hr][2][2] = gr9[hr]
    print(z)


def find_latlon():
    lat_delta = 0.005 / 2
    lon_delta = 0.005 / 2
    latlon = list(df.columns.values)
    for i in latlon:
        if '_' in i:
            lon = i.split('_', 1)[0]
            lat = i.split('_', 1)[1]
            top = float(lon) + lon_delta
            down = float(lon) - lon_delta
            right = float(lat) + lat_delta
            left = float(lat) - lat_delta
            zoom_lat = Interval(right, left)
            zoom_lon = Interval(top, down)
            lon_711 = 121.55130754335269
            lat_711 = 25.03977333344578
            # 25.017034677583894, 121.53373092305938
            lon_bol = lon_711 in zoom_lon
            lat_bol = lat_711 in zoom_lat
            if lon_bol == True and lat_bol == True:
                print(i)


def test_data_2019():
    df2019 = pd.read_csv('./test_2019.csv')
    print(df2019)
    col_grid = df2019.columns
    col_grid = col_grid.drop('Time')
    for col in col_grid:
        print(col)
        grid_data = pd.DataFrame()
        grid_data['TIME'] = df2019['Time']
        grid_data['POPULATION'] = df2019[col]
        print(grid_data.isnull().any().sum())
        print(grid_data.isna().any().sum())


def save_z(lon, lat, ty):
    delta = 0.005
    right = str('%.03f' % (lon + delta))
    left = str('%.03f' % (lon - delta))
    top = str('%.03f' % (lat + delta))
    down = str('%.03f' % (lat - delta))
    lon = str('%.03f' % lon)
    lat = str('%.03f' % lat)

    left_top, top, right_top = f'{left}_{top}', f'{lon}_{top}', f'{right}_{top}'
    left_center, center, right_center = f'{left}_{lat}', f'{lon}_{lat}', f'{right}_{lat}'
    left_down, down, right_down = f'{left}_{down}', f'{lon}_{down}', f'{right}_{down}'
    loc_grid_list = [left_top, top, right_top, left_center, center, right_center, left_down, down, right_down]

    z = np.zeros((8760, 3, 3))
    index_1 = 0
    index_2 = 0
    for loc in loc_grid_list:
        gr = pd.read_csv(f'./grid_data_{ty}/{str(loc)}.csv')
        for hr in range(8760):
            z[hr][index_1][index_2] = gr.POPULATION[hr]
        index_2 += 1
        if index_2 == 3:
            index_2 = 0
            index_1 += 1
    np.save(f'./{location}/save_z_{ty}.npy', z)


def test_target(n, lon, lat):
    if not os.path.exists(f'./{location}/train_target/{n}sig'):
        os.makedirs(f'./{location}/train_target/{n}sig')
    if not os.path.exists(f'./{location}/test_target/{n}sig'):
        os.makedirs(f'./{location}/test_target/{n}sig')
    delta = 0.005
    right = str('%.03f' % (lon + delta))
    left = str('%.03f' % (lon - delta))
    top = str('%.03f' % (lat + delta))
    down = str('%.03f' % (lat - delta))
    lon = str('%.03f' % lon)
    lat = str('%.03f' % lat)
    print(right, left, top, down)
    left_top, top, right_top = f'{left}_{top}', f'{lon}_{top}', f'{right}_{top}'
    left_center, center, right_center = f'{left}_{lat}', f'{lon}_{lat}', f'{right}_{lat}'
    left_down, down, right_down = f'{left}_{down}', f'{lon}_{down}', f'{right}_{down}'
    loc_grid_list = [left_top, top, right_top, left_center, center, right_center, left_down, down, right_down]

    train_threshold = pd.read_csv(f'./{location}/{n}sig/threshold_{str(center)}.csv')

    df_all_train = pd.DataFrame()
    df_all_test = pd.DataFrame()

    for k, loc in enumerate(loc_grid_list):
        test_target = pd.DataFrame()
        train_target = pd.DataFrame()
        gr_train = pd.read_csv(f'./grid_data_train/{str(loc)}.csv')
        gr_test = pd.read_csv(f'./grid_data_test/{str(loc)}.csv')
        threshold = train_threshold.at[k, 'threshold']

        for i in range(len(gr_test)):
            pop_train = gr_train.at[i, 'POPULATION']
            if pop_train > threshold:
                train_target.at[i, 'target'] = 1
            else:
                train_target.at[i, 'target'] = 0

            pop_test = gr_test.at[i, 'POPULATION']
            if pop_test > threshold:
                test_target.at[i, 'target'] = 1
            else:
                test_target.at[i, 'target'] = 0

        train_target.to_csv(f'./{location}/train_target/{n}sig/{str(loc)}.csv', index=False, encoding='utf-8-sig')
        test_target.to_csv(f'./{location}/test_target/{n}sig/{str(loc)}.csv', index=False, encoding='utf-8-sig')
        df_all_train[loc] = train_target['target']
        df_all_test[loc] = test_target['target']
    print(df_all_train)
    print(df_all_test)


    df_all_train['Sum'] = df_all_train.apply(lambda x: x.sum(), axis=1)
    df_all_test['Sum'] = df_all_test.apply(lambda x: x.sum(), axis=1)

    df_all_train[df_all_train > 1] = 1
    df_all_test[df_all_test > 1] = 1
    df_all_train.to_csv(f'./{location}/train_target/{n}sig/all.csv', index=False, encoding='utf-8-sig')
    df_all_test.to_csv(f'./{location}/test_target/{n}sig/all.csv', index=False, encoding='utf-8-sig')
    print(df_all_train)
    print(df_all_test)


def test_real():
    lon, lat = 121.520, 25.050
    delta = 0.005
    right = str('%.03f' % (lon + delta))
    left = str('%.03f' % (lon - delta))
    top = str('%.03f' % (lat + delta))
    down = str('%.03f' % (lat - delta))
    lon = str('%.03f' % lon)
    lat = str('%.03f' % lat)
    print(right, left, top, down)
    left_top, top, right_top = f'{left}_{top}', f'{lon}_{top}', f'{right}_{top}'
    left_center, center, right_center = f'{left}_{lat}', f'{lon}_{lat}', f'{right}_{lat}'
    left_down, down, right_down = f'{left}_{down}', f'{lon}_{down}', f'{right}_{down}'
    location_list = ['left_top', 'top', 'right_top', 'left', 'center', 'right', 'left_down', 'down', 'right_down']
    loc_grid_list = [left_top, top, right_top, left_center, center, right_center, left_down, down, right_down]

    real_data = pd.DataFrame()
    real_data['grid1_real'] = test1.POPULATION
    real_data['grid2_real'] = test2.POPULATION
    real_data['grid3_real'] = test3.POPULATION
    real_data['grid4_real'] = test4.POPULATION
    real_data['grid5_real'] = test5.POPULATION
    real_data['grid6_real'] = test6.POPULATION
    real_data['grid7_real'] = test7.POPULATION
    real_data['grid8_real'] = test8.POPULATION
    real_data['grid9_real'] = test9.POPULATION

    for i in range(len(real_data)):
        pop_sum = real_data.at[i, 'grid1_real'] + real_data.at[i, 'grid2_real'] + real_data.at[i, 'grid3_real'] + \
                  real_data.at[i, 'grid4_real'] + real_data.at[i, 'grid5_real'] + real_data.at[i, 'grid6_real'] + \
                  real_data.at[i, 'grid7_real'] + real_data.at[i, 'grid8_real'] + real_data.at[i, 'grid9_real']
        real_data.at[i, 'Mean'] = pop_sum / 9


def main():
    lon, lat = 121.565, 25.035  # Taipei101

    for n in [1.5, 2, 3]:
        sigma_create_data(n, lon, lat)
        save_z(lon, lat, 'train')  # 8760*3*3
        save_z(lon, lat, 'test')
        test_target(n, lon, lat)  # 8760*1

main()
