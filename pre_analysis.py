import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
import keras
from numpy import array


def OnlyCharNum(s):
    s2 = s.lower()
    fomart = 'abcdefghijklmnopqrstuvwxyz0123456789:'
    for c in s2:
        if not c in fomart:
            s = s.replace(c, '')
            if '-' in s:
                s = s.replace('-', '_')
                if ':' in s:
                    s = s.replace(':', '_')

                return s


def pre_one_time_9grid(location, sig_n, long, pre, test):

    grid_list = ['left_top', 'top', 'right_top', 'left', 'center', 'right', 'left_down', 'down', 'right_down']

    result_cnn_D_emsemble_longk = pd.read_csv(
        f'{location}/{sig_n}sig/result_cnn_D_emsemble_long{long}.csv').reset_index()
    people_grid = pd.read_csv(f'{location}/{sig_n}sig/result_emsemble.csv')
    start = look_back + long - 1
    print(result_cnn_D_emsemble_longk.columns)
    print(start)
    col_TIME = people_grid['TIME'].iloc[start:].reset_index(drop=True).reset_index()
    print(result_cnn_D_emsemble_longk)
    temp, gr1, gr2, gr3, gr4, gr5, gr6, gr7, gr8, gr9 = cnn_predict()
    pre_data = [gr1, gr2, gr3, gr4, gr5, gr6, gr7, gr8, gr9]

    rule_pre = result_cnn_D_emsemble_longk['predict_y'] == pre
    rule_test = result_cnn_D_emsemble_longk['test_y'] == test
    analysis_data = result_cnn_D_emsemble_longk[rule_pre & rule_test].reset_index(drop=True)
    times = ['t-' + str(long + 1), 't-' + str(long), 't-' + str(long - 1), 't+1']
    color_scale = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF']

    for i in range(len(analysis_data)):  # len(analysis_data)
        TIME = analysis_data.at[i, 'TIME']
        index = analysis_data.at[i, 'index']
        row_index = int(people_grid[people_grid['TIME'] == TIME].index.values)
        fig = go.Figure()
        past_pop = pd.DataFrame()
        for g, grid in enumerate(grid_list):
            past_pop.at[g, 'location'] = grid
            grid_col = str(grid) + '_POP'
            # print(row_index - long, index)
            pre_value = [None, None, people_grid.at[row_index - long, grid_col], pre_data[g][index]]
            value = []
            for t, time in enumerate(times):
                past_pop.at[g, time] = people_grid.at[row_index - long - 2 + t, grid_col]
                if t == 3:
                    past_pop.at[g, time] = people_grid.at[row_index, grid_col]

                value.append(past_pop.at[g, time])
            # print(value)
            # print(past_pop.iloc[g])

            fig.add_trace(go.Scatter(x=times, y=value,
                                     mode='lines+markers',
                                     name=grid,
                                     line=dict(color=color_scale[g], width=4.5),
                                     marker=dict(size=10))

                          )

            fig.add_trace(go.Scatter(x=times, y=pre_value,
                                     mode='lines+markers',
                                     name=f'{grid}_pre',
                                     line=dict(color=color_scale[g], width=5, dash='dash'),
                                     marker=dict(size=10)
                                     ))
        # print(past_pop)
        if location == 'Chungsiou':
            location = 'Zhongxiao'

        elif location == 'Taipei101':
            location = 'Xinyi'

        fig.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'size': 52, 'family': 'Times New Roman'},

            title=dict(text=f'{location} Time:{TIME}',
                       xanchor='center',
                       x=0.5, ),

            yaxis=dict(title_text='Population',
                       tickfont=dict(size=55), ),
            xaxis=dict(title='Time(hour)',
                       tickfont=dict(size=55)))
        # print(TIME)
        if location == 'Zhongxiao':
            location = 'Chungsiou'

        elif location == 'Xinyi':
            location = 'Taipei101'

        weekday = pd.Timestamp(TIME).weekday()
        # print(weekday)
        if weekday == 0 or weekday == 1 or weekday == 2 or weekday == 3 or weekday == 4:
            TIME = OnlyCharNum(TIME)
            # print(TIME)
            fig.write_image(f'{location}/{sig_n}sig/pic_one_time_9grid/pre{pre}test{test}/normal_day/'
                            f'plot_time{TIME}.png',
                            width=1920, height=1080)
        elif weekday == 5 or weekday == 6:
            TIME = OnlyCharNum(TIME)
            # print(TIME)
            fig.write_image(f'{location}/{sig_n}sig/pic_one_time_9grid/pre{pre}test{test}/holiday_day/'
                            f'plot_time{TIME}.png',
                            width=1920, height=1080)
        # fig.show()


def create_dataset(temp):
    # print(temp.shape)
    X, y = list(), list()
    for i in range(len(temp)):  # len(train)
        x_end = i + look_back
        y_end = i + look_back + long - 1
        if y_end > len(temp) - 1:
            break
        temp_x, temp_y = temp[i:x_end, :, :], temp[y_end]
        # print(temp_x.shape)
        temp_x = temp_x.reshape(3, 3, 3)

        X.append(temp_x)
        y.append(temp_y)

    return array(X), array(y)


def nor(data_2018, data_2019):
    temp = data_2018.reshape(-1, 1).tolist()
    train_max = max(map(max, temp))
    train_max_half = train_max / 2
    data_2018_v2 = (data_2018.astype(np.float32) - train_max_half) / train_max_half
    data_2019_v2 = (data_2019.astype(np.float32) - train_max_half) / train_max_half

    train = data_2018_v2[:, :, :, None]  # (8760, 3, 3, 1)
    test = data_2019_v2[:, :, :, None]  # (8760, 3, 3, 1)
    # print(train.shape)
    # print(test.shape)

    return train, test


def create_timestamp_X_Y_cnn(data_2018, data_2019):
    train, test = nor(data_2018, data_2019)
    train_X, train_y = create_dataset(train)
    test_X, test_y = create_dataset(test)

    return train_X, train_y, test_X, test_y


def cnn_predict():
    data_2018 = np.load(f'./{location}/save_z_train.npy')
    data_2019 = np.load(f'./{location}/save_z_test.npy')
    train_X, train_y, test_X, test_y = create_timestamp_X_Y_cnn(data_2018, data_2019)
    cnn_model = keras.models.load_model(f'./{location}/cnn_model_long{long}.h5')
    pre_test = cnn_model.predict(test_X)

    temp = data_2018.reshape(-1, 1).tolist()
    X_max = max(map(max, temp))
    X_max_half = X_max / 2
    pre = pre_test * X_max_half + X_max_half
    print(pre)

    gr1, gr2, gr3, gr4, gr5, gr6, gr7, gr8, gr9 = [], [], [], [], [], [], [], [], []
    time_len = 8760 - look_back - long + 1
    z = np.zeros((time_len, 3, 3))
    for hr in range(time_len):
        gr1.append(pre[hr][0][0][0])
        gr2.append(pre[hr][0][1][0])
        gr3.append(pre[hr][0][2][0])
        gr4.append(pre[hr][1][0][0])
        gr5.append(pre[hr][1][1][0])
        gr6.append(pre[hr][1][2][0])
        gr7.append(pre[hr][2][0][0])
        gr8.append(pre[hr][2][1][0])
        gr9.append(pre[hr][2][2][0])

    return pre, gr1, gr2, gr3, gr4, gr5, gr6, gr7, gr8, gr9


location = 'Taipei101'
sig_n = 2
long = 2
look_back = 3

def run_pre_one_time_9grid():
    for p in range(0, 2):
        for t in range(0, 2):
            print((p, t))
            # print(p)
            # print(t)
            if p == 0 and t == 0:
                continue
            else:
                pre = p
                test = t
                pre_one_time_9grid(location, sig_n, long, pre, test)


run_pre_one_time_9grid()


def pre_all_time_9grid(location, sig_n, long, pre, test):
    grid_list = ['left_top', 'top', 'right_top', 'left', 'center', 'right', 'left_down', 'down', 'right_down']

    for g in grid_list:
        grid_location = g
        result_cnn_D_emsemble_longk = pd.read_csv(f'{location}/{sig_n}sig/result_cnn_D_emsemble_long{long}.csv')
        people_grid = pd.read_csv(f'{location}/{sig_n}sig/result_emsemble.csv')
        # print(result_cnn_D_emsemble_longk.iloc[0])

        rule_pre = result_cnn_D_emsemble_longk['predict_y'] == pre
        rule_test = result_cnn_D_emsemble_longk['test_y'] == test
        analysis_data = result_cnn_D_emsemble_longk[rule_pre & rule_test].reset_index(drop=True)

        past_pop = pd.DataFrame()

        for i in range(len(analysis_data)):
            TIME = analysis_data.at[i, 'TIME']
            row_index = people_grid[people_grid['TIME'] == TIME].index.values
            time_index_0 = int(row_index)
            time_index_1 = int(row_index - long)
            time_index_2 = int(row_index - long - 1)
            time_index_3 = int(row_index - long - 2)

            grid_col = str(grid_location) + '_POP'
            grid1_pop_0 = people_grid.at[time_index_0, grid_col]
            grid1_pop_1 = people_grid.at[time_index_1, grid_col]
            grid1_pop_2 = people_grid.at[time_index_2, grid_col]
            grid1_pop_3 = people_grid.at[time_index_3, grid_col]

            past_pop.at[3, 'time'] = 't+1'
            past_pop.at[2, 'time'] = 't-' + str(long - 1)
            past_pop.at[1, 'time'] = 't-' + str(long)
            past_pop.at[0, 'time'] = 't-' + str(long + 1)

            past_pop.at[3, str(TIME)] = grid1_pop_0
            past_pop.at[2, str(TIME)] = grid1_pop_1
            past_pop.at[1, str(TIME)] = grid1_pop_2
            past_pop.at[0, str(TIME)] = grid1_pop_3
            past_pop = past_pop.reset_index(drop=True)
        # print(past_pop)

        fig = px.line(past_pop, x="time", y=past_pop.columns,
                      title=f'Time Series with population\npre:{pre}, test:{test}\n/threshold:{sig_n}sigma\n/long={long}\n'
                            f' location:{location}/grid:{grid_location}')

        fig.update_xaxes(
            dtick="M1",
            tickformat="%b\n%Y",
            ticklabelmode="period",
            rangeslider_visible=False)
        fig.update_layout(
            autosize=True,
            width=1200,
            height=800)
        fig.show()
