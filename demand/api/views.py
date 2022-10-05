from rest_framework.viewsets import ModelViewSet
from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil import relativedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.layers import Dense
import plotly.offline as pyoff
import plotly.graph_objects as go
import plotly.io as pio
import os
# from rest_framework.permissions import IsAuthenticatedOrReadOnly


from demand.models import Demand
from demand.api.serializers import DemandSerializer


class DemandApiViewSet(ModelViewSet):
    # permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = DemandSerializer
    # queryset = Demand.objects.all()


class PredictMonthView(APIView):
    def get(self, request, month):
        print("REQUEST_TEST", month)
        # serializer = DemandSerializer(request.demand)
        df = pd.read_csv(os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'dataset/dataset-degala-mes-2.csv'))
        print(df)
        # Eliminar columnas que no usaremos y cambiar de nombre 'datesold' y 'price' por 'mes' y 'ventas'.
        df = df.rename(columns={'date': 'mes', 'price': 'ventas'})

        # Agrupar meses, sumando las ventas totales del mes.
        # df = df.groupby('mes')['ventas'].agg('sum')
        df = df.groupby('mes')['ventas'].agg('sum')
        df = df.reset_index()

        def next_month_predict(df, cantidad_meses):
            df_final = pd.DataFrame()
            for k in range(cantidad_meses):
                next_month = (datetime.strptime(str(list(
                    df['mes'])[-1]), '%Y-%m') + relativedelta.relativedelta(months=1)).strftime('%Y-%m')
                df = df.append(pd.DataFrame(
                    data={'mes': [next_month], 'ventas': [0]}))

                df_diff = df.copy()
                df_diff['prev_ventas'] = df_diff['ventas'].shift(1)
                df_diff = df_diff.dropna()
                df_diff['diff'] = (df_diff['ventas'] - df_diff['prev_ventas'])
                df_supervised = df_diff.drop(['prev_ventas'], axis=1)

                for inc in range(1, 13):
                    field_name = 'lag_' + str(inc)
                    df_supervised[field_name] = df_supervised['diff'].shift(
                        inc)

                    df_supervised = df_supervised.dropna().reset_index(drop=True)
                    df_model = df_supervised.drop(['ventas', 'mes'], axis=1)
                    train_set, test_set = df_model[:-
                                                   1].values, df_model[-1:].values

                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaler = scaler.fit(train_set)

                    train_set = train_set.reshape(
                        train_set.shape[0], train_set.shape[1])
                    train_set_scaled = scaler.transform(train_set)
                    test_set = test_set.reshape(
                        test_set.shape[0], test_set.shape[1])
                test_set_scaled = scaler.transform(test_set)

                X_train, y_train = train_set_scaled[:,
                                                    1:], train_set_scaled[:, 0:1]
                X_train = X_train.reshape(
                    X_train.shape[0], 1, X_train.shape[1])
                X_test, y_test = test_set_scaled[:,
                                                 1:], test_set_scaled[:, 0:1]
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

                model = Sequential()
                model.add(LSTM(4, batch_input_shape=(
                    1, X_train.shape[1], X_train.shape[2]), stateful=True))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(X_train, y_train, epochs=12,
                          batch_size=1, verbose=1, shuffle=False)

                y_pred = model.predict(X_test, batch_size=1)
                y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

                pred_test_set = []
                for index in range(0, len(y_pred)):
                    pred_test_set.append(np.concatenate(
                        [y_pred[index], X_test[index]], axis=1))

                pred_test_set = np.array(pred_test_set)
                pred_test_set = pred_test_set.reshape(
                    pred_test_set.shape[0], pred_test_set.shape[2])
                pred_test_set_inverted = scaler.inverse_transform(
                    pred_test_set)

                result_list = []
                sales_dates = list(df[-2:].mes)
                act_sales = list(df[-2:].ventas)

                result_dic = {}
                result_dic['pred_value'] = int(act_sales[0])
                result_dic['mes'] = sales_dates[0]
                result_list.append(result_dic)

                for index in range(0, len(pred_test_set_inverted)):
                    result_dict = {}
                    result_dict['pred_value'] = int(
                        pred_test_set_inverted[index][0] + act_sales[index])
                    result_dict['mes'] = sales_dates[index+1]
                    result_list.append(result_dict)

                df_result = pd.DataFrame(result_list)
                if k == 0:
                    df_final['pred_value'] = list(df_result['pred_value'])
                    df_final['mes'] = list(df_result['mes'])
                else:
                    df_final = df_final.append(pd.DataFrame(data={'pred_value': [list(
                        df_result['pred_value'])[-1]], 'mes': [list(df_result['mes'])[-1]]}))

                df = df[:-1]
                df = df.append(pd.DataFrame(
                    data={'mes': [next_month], 'ventas': [list(df_result['pred_value'])[-1]]}))
                df = df.reset_index().drop('index', axis=1)

            # Graficamos los resultados.
            df_sales_pred = pd.merge(df, df_final, on='mes', how='left')
            df_sales_pred = df_sales_pred.fillna('')
            return df_sales_pred
        df_next_month = next_month_predict(df, month)
        return Response(df_next_month)


class PredictView(APIView):
    def get(self, request):
        # serializer = DemandSerializer(request.demand)
        df = pd.read_csv(os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'dataset/dataset-degala-mes-2.csv'))
        print(df)
        # Eliminar columnas que no usaremos y cambiar de nombre 'datesold' y 'price' por 'mes' y 'ventas'.
        df = df.rename(columns={'date': 'mes', 'price': 'ventas'})

        # Agrupar meses, sumando las ventas totales del mes.
        # df = df.groupby('mes')['ventas'].agg('sum')
        df = df.groupby('mes')['ventas'].agg('sum')
        df = df.reset_index()

        df_diff = df.copy()
        df_diff['prev_ventas'] = df_diff['ventas'].shift(1)
        df_diff = df_diff.dropna()
        df_diff['diff'] = (df_diff['ventas'] - df_diff['prev_ventas'])
        df_diff

        df_supervised = df_diff.drop(['prev_ventas'], axis=1)

        for inc in range(1, 13):
            field_name = 'lag_' + str(inc)
            df_supervised[field_name] = df_supervised['diff'].shift(inc)

        df_supervised = df_supervised.dropna().reset_index(drop=True)
        df_supervised
        df_model = df_supervised.drop(['ventas', 'mes'], axis=1)
        train_set, test_set = df_model[0:-12].values, df_model[-12:].values

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(train_set)

        train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
        train_set_scaled = scaler.transform(train_set)

        test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
        test_set_scaled = scaler.transform(test_set)

        X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Ajustando el modelo
        model = Sequential()
        model.add(LSTM(4, batch_input_shape=(
            1, X_train.shape[1], X_train.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=12,
                  batch_size=1, verbose=1, shuffle=False)

        # Predicción del modelo.
        y_pred = model.predict(X_test, batch_size=1)

        # Transformación inversa del escalamiento.
        y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

        pred_test_set = []
        for index in range(0, len(y_pred)):
            pred_test_set.append(np.concatenate(
                [y_pred[index], X_test[index]], axis=1))

        pred_test_set = np.array(pred_test_set)
        pred_test_set = pred_test_set.reshape(
            pred_test_set.shape[0], pred_test_set.shape[2])
        pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

        # Transformación inversa de las diferencias de ventas.
        result_list = []
        sales_dates = list(df[-13:].mes)
        act_sales = list(df[-13:].ventas)
        for index in range(0, len(pred_test_set_inverted)):
            result_dict = {}
            result_dict['mes'] = sales_dates[index+1]
            result_dict['pred_value'] = int(
                pred_test_set_inverted[index][0] + act_sales[index])
            result_dict['real_value'] = int(act_sales[index+1])
            result_list.append(result_dict)

        df_result = pd.DataFrame(result_list)
        df_result

        df_sales_pred = pd.merge(df, df_result, on='mes', how='left')
        df_sales_pred = df_sales_pred.fillna('')

        return Response(df_sales_pred)
