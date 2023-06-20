# from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.viewsets import ModelViewSet
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.views import APIView
from rest_framework.response import Response

import pandas as pd
import numpy as np

from datetime import datetime
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

from dataset.models import DataSet
from demand.models import Demand
from demand.api.serializers import DemandSerializer


class DemandApiViewSet(ModelViewSet):
    # permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = DemandSerializer
    queryset = Demand.objects.all()
    filter_backends = [DjangoFilterBackend]
    filterset_fields = {"date": ["range"]}


# class PredictMonthView(APIView):
#     def get(self, request, month):
#         # serializer = DemandSerializer(request.demand)
#         df = pd.read_csv(
#             os.path.join(
#                 os.path.dirname(os.path.dirname(__file__)),
#                 "dataset/dataset-degala-mes-2.csv",
#             )
#         )
#         print(df)
#         # Eliminar columnas que no usaremos y cambiar de nombre 'datesold' y 'price' por 'mes' y 'ventas'.
#         df = df.rename(columns={"date": "mes", "price": "ventas"})

#         # Agrupar meses, sumando las ventas totales del mes.
#         # df = df.groupby('mes')['ventas'].agg('sum')
#         df = df.groupby("mes")["ventas"].agg("sum")
#         df = df.reset_index()

#         def next_month_predict(df, cantidad_meses):
#             df_final = pd.DataFrame()
#             for k in range(cantidad_meses):
#                 next_month = (
#                     datetime.strptime(str(list(df["mes"])[-1]), "%Y-%m")
#                     + relativedelta.relativedelta(months=1)
#                 ).strftime("%Y-%m")
#                 df = df.append(pd.DataFrame(data={"mes": [next_month], "ventas": [0]}))

#                 df_diff = df.copy()
#                 df_diff["prev_ventas"] = df_diff["ventas"].shift(1)
#                 df_diff = df_diff.dropna()
#                 df_diff["diff"] = df_diff["ventas"] - df_diff["prev_ventas"]
#                 df_supervised = df_diff.drop(["prev_ventas"], axis=1)

#                 for inc in range(1, 13):
#                     field_name = "lag_" + str(inc)
#                     df_supervised[field_name] = df_supervised["diff"].shift(inc)

#                     df_supervised = df_supervised.dropna().reset_index(drop=True)
#                     df_model = df_supervised.drop(["ventas", "mes"], axis=1)
#                     train_set, test_set = df_model[:-1].values, df_model[-1:].values

#                     scaler = MinMaxScaler(feature_range=(-1, 1))
#                     scaler = scaler.fit(train_set)

#                     train_set = train_set.reshape(
#                         train_set.shape[0], train_set.shape[1]
#                     )
#                     train_set_scaled = scaler.transform(train_set)
#                     test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
#                 test_set_scaled = scaler.transform(test_set)

#                 X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
#                 X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
#                 X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
#                 X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#                 model = Sequential()
#                 model.add(
#                     LSTM(
#                         4,
#                         batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
#                         stateful=True,
#                     )
#                 )
#                 model.add(Dense(1))
#                 model.compile(loss="mean_squared_error", optimizer="adam")
#                 model.fit(
#                     X_train, y_train, epochs=12, batch_size=1, verbose=1, shuffle=False
#                 )

#                 y_pred = model.predict(X_test, batch_size=1)
#                 y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

#                 pred_test_set = []
#                 for index in range(0, len(y_pred)):
#                     pred_test_set.append(
#                         np.concatenate([y_pred[index], X_test[index]], axis=1)
#                     )

#                 pred_test_set = np.array(pred_test_set)
#                 pred_test_set = pred_test_set.reshape(
#                     pred_test_set.shape[0], pred_test_set.shape[2]
#                 )
#                 pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

#                 result_list = []
#                 sales_dates = list(df[-2:].mes)
#                 act_sales = list(df[-2:].ventas)

#                 result_dic = {}
#                 result_dic["pred_value"] = int(act_sales[0])
#                 result_dic["mes"] = sales_dates[0]
#                 result_list.append(result_dic)

#                 for index in range(0, len(pred_test_set_inverted)):
#                     result_dict = {}
#                     result_dict["pred_value"] = int(
#                         pred_test_set_inverted[index][0] + act_sales[index]
#                     )
#                     result_dict["mes"] = sales_dates[index + 1]
#                     result_list.append(result_dict)

#                 df_result = pd.DataFrame(result_list)
#                 if k == 0:
#                     df_final["pred_value"] = list(df_result["pred_value"])
#                     df_final["mes"] = list(df_result["mes"])
#                 else:
#                     df_final = df_final.append(
#                         pd.DataFrame(
#                             data={
#                                 "pred_value": [list(df_result["pred_value"])[-1]],
#                                 "mes": [list(df_result["mes"])[-1]],
#                             }
#                         )
#                     )

#                 df = df[:-1]
#                 df = df.append(
#                     pd.DataFrame(
#                         data={
#                             "mes": [next_month],
#                             "ventas": [list(df_result["pred_value"])[-1]],
#                         }
#                     )
#                 )
#                 df = df.reset_index().drop("index", axis=1)

#             # Graficamos los resultados.
#             df_sales_pred = pd.merge(df, df_final, on="mes", how="left")
#             df_sales_pred = df_sales_pred.fillna("")
#             return df_sales_pred

#         df_next_month = next_month_predict(df, month)
#         return Response(df_next_month)


class PredictMonthView(APIView):
    def get(self, request, month):
        # serializer = DemandSerializer(request.demand)
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "dataset/dataset-degala-mes-2.csv",
            )
        )
        print(df)
        df = df.rename(columns={"date": "mes", "price": "ventas"})
        df = df.groupby("mes")["ventas"].agg("sum")
        df = df.reset_index()

        def next_month_predict(df, cantidad_meses):
            df_final = pd.DataFrame()
            for k in range(cantidad_meses):
                next_month = (
                    datetime.strptime(str(list(df["mes"])[-1]), "%Y-%m")
                    + relativedelta.relativedelta(months=1)
                ).strftime("%Y-%m")
                df = df.append(pd.DataFrame(data={"mes": [next_month], "ventas": [0]}))

                df_diff = df.copy()
                df_diff["prev_ventas"] = df_diff["ventas"].shift(1)
                df_diff = df_diff.dropna()
                df_diff["diff"] = df_diff["ventas"] - df_diff["prev_ventas"]
                df_supervised = df_diff.drop(["prev_ventas"], axis=1)

                for inc in range(1, 13):
                    field_name = "lag_" + str(inc)
                    df_supervised[field_name] = df_supervised["diff"].shift(inc)

                    df_supervised = df_supervised.dropna().reset_index(drop=True)
                    df_model = df_supervised.drop(["ventas", "mes"], axis=1)
                    train_set, test_set = df_model[:-1].values, df_model[-1:].values

                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaler = scaler.fit(train_set)

                    train_set = train_set.reshape(
                        train_set.shape[0], train_set.shape[1]
                    )
                    train_set_scaled = scaler.transform(train_set)
                    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
                test_set_scaled = scaler.transform(test_set)

                X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

                model = Sequential()
                model.add(
                    LSTM(
                        4,
                        batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                        stateful=True,
                    )
                )
                model.add(Dense(1))
                model.compile(loss="mean_squared_error", optimizer="adam")
                model.fit(
                    X_train, y_train, epochs=12, batch_size=1, verbose=1, shuffle=False
                )

                y_pred = model.predict(X_test, batch_size=1)
                y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

                pred_test_set = []
                for index in range(0, len(y_pred)):
                    pred_test_set.append(
                        np.concatenate([y_pred[index], X_test[index]], axis=1)
                    )

                pred_test_set = np.array(pred_test_set)
                pred_test_set = pred_test_set.reshape(
                    pred_test_set.shape[0], pred_test_set.shape[2]
                )
                pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

                result_list = []
                sales_dates = list(df[-2:].mes)
                act_sales = list(df[-2:].ventas)

                result_dic = {}
                result_dic["pred_value"] = int(act_sales[0])
                result_dic["mes"] = sales_dates[0]
                result_list.append(result_dic)

                for index in range(0, len(pred_test_set_inverted)):
                    result_dict = {}
                    result_dict["pred_value"] = int(
                        pred_test_set_inverted[index][0] + act_sales[index]
                    )
                    result_dict["mes"] = sales_dates[index + 1]
                    result_list.append(result_dict)

                df_result = pd.DataFrame(result_list)
                if k == 0:
                    df_final["pred_value"] = list(df_result["pred_value"])
                    df_final["mes"] = list(df_result["mes"])
                else:
                    df_final = df_final.append(
                        pd.DataFrame(
                            data={
                                "pred_value": [list(df_result["pred_value"])[-1]],
                                "mes": [list(df_result["mes"])[-1]],
                            }
                        )
                    )

                df = df[:-1]
                df = df.append(
                    pd.DataFrame(
                        data={
                            "mes": [next_month],
                            "ventas": [list(df_result["pred_value"])[-1]],
                        }
                    )
                )
                df = df.reset_index().drop("index", axis=1)

            # Graficamos los resultados.
            df_sales_pred = pd.merge(df, df_final, on="mes", how="left")
            df_sales_pred = df_sales_pred.fillna("")
            return df_sales_pred

        df_next_month = next_month_predict(df, month)
        return Response(df_next_month)


class PredictView(APIView):
    def get(self, request):
        # serializer = DemandSerializer(request.demand)
        df = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "dataset/dataset-degala-mes-2.csv",
            )
        )
        print(df)
        # Eliminar columnas que no usaremos y cambiar de nombre 'datesold' y 'price' por 'mes' y 'ventas'.
        df = df.rename(columns={"date": "mes", "price": "ventas"})

        # Agrupar meses, sumando las ventas totales del mes.
        # df = df.groupby('mes')['ventas'].agg('sum')
        df = df.groupby("mes")["ventas"].agg("sum")
        df = df.reset_index()

        df_diff = df.copy()
        df_diff["prev_ventas"] = df_diff["ventas"].shift(1)
        df_diff = df_diff.dropna()
        df_diff["diff"] = df_diff["ventas"] - df_diff["prev_ventas"]
        df_diff

        df_supervised = df_diff.drop(["prev_ventas"], axis=1)

        for inc in range(1, 13):
            field_name = "lag_" + str(inc)
            df_supervised[field_name] = df_supervised["diff"].shift(inc)

        df_supervised = df_supervised.dropna().reset_index(drop=True)
        df_supervised
        df_model = df_supervised.drop(["ventas", "mes"], axis=1)
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
        model.add(
            LSTM(
                4,
                batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                stateful=True,
            )
        )
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(X_train, y_train, epochs=12, batch_size=1, verbose=1, shuffle=False)

        # Predicción del modelo.
        y_pred = model.predict(X_test, batch_size=1)

        # Transformación inversa del escalamiento.
        y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

        pred_test_set = []
        for index in range(0, len(y_pred)):
            pred_test_set.append(np.concatenate([y_pred[index], X_test[index]], axis=1))

        pred_test_set = np.array(pred_test_set)
        pred_test_set = pred_test_set.reshape(
            pred_test_set.shape[0], pred_test_set.shape[2]
        )
        pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

        # Transformación inversa de las diferencias de ventas.
        result_list = []
        sales_dates = list(df[-13:].mes)
        act_sales = list(df[-13:].ventas)
        for index in range(0, len(pred_test_set_inverted)):
            result_dict = {}
            result_dict["mes"] = sales_dates[index + 1]
            result_dict["pred_value"] = int(
                pred_test_set_inverted[index][0] + act_sales[index]
            )
            result_dict["real_value"] = int(act_sales[index + 1])
            result_list.append(result_dict)

        df_result = pd.DataFrame(result_list)
        df_result

        df_sales_pred = pd.merge(df, df_result, on="mes", how="left")
        df_sales_pred = df_sales_pred.fillna("")

        return Response(df_sales_pred)


class DemandPredictMonthView(APIView):
    def get(self, request):
        from_date_param = request.query_params.get(
            "from_date", None
        )  # FORMATO: YYYY-MM
        to_date_param = request.query_params.get("to_date", None)  # FORMATO: YYYY-MM
        from_df = Demand.objects.filter(date=from_date_param)
        to_df = Demand.objects.filter(date=to_date_param)
        if len(from_df) == 0:
            response_data = predict_demand_by_months_from_dataset_or_demand_with_model(
                from_date_param, to_date_param
            )
            return Response(response_data)
        elif len(from_df) > 0 and len(to_df) == 0:
            response_data = predict_demand_by_months_from_date_with_model(
                from_date_param, to_date_param
            )
            return Response(response_data)
        elif len(from_df) > 0 and len(to_df) > 0:
            response_data = predict_demand_by_months_from_date_to_date(
                from_date_param, to_date_param
            )
            return Response(response_data)
        else:
            return Response([])


class DemandPredictYearView(APIView):
    def get(self, request):
        from_date_param = request.query_params.get(
            "from_date", None
        )  # FORMATO: YYYY-MM
        to_date_param = request.query_params.get("to_date", None)  # FORMATO: YYYY-MM
        from_df = Demand.objects.filter(date=from_date_param)
        to_df = Demand.objects.filter(date=to_date_param)
        if len(from_df) == 0:
            response_data = predict_demand_by_months_from_dataset_or_demand_with_model(
                from_date_param, to_date_param
            )
            return Response(response_data)
        elif len(from_df) > 0 and len(to_df) == 0:
            response_data = predict_demand_by_months_from_date_with_model(
                from_date_param, to_date_param
            )
            return Response(response_data)
        elif len(from_df) > 0 and len(to_df) > 0:
            response_data = predict_demand_by_months_from_date_to_date(
                from_date_param, to_date_param
            )
            return Response(response_data)
        else:
            return Response([])


def predict_demand_by_months_from_dataset_or_demand_with_model(
    from_date_param, to_date_param
):
    queryset_demand = Demand.objects.all()
    queryset_dataset = DataSet.objects.all()

    df_demand = pd.DataFrame.from_records(queryset_demand.values())
    df_dataset = pd.DataFrame.from_records(queryset_dataset.values())

    last_date_demand = df_demand["date"].max()
    last_date_dataset = df_dataset["date"].max()

    df = df_dataset

    if last_date_demand > last_date_dataset:
        df = df_demand
    else:
        df = df_dataset

    df["sales"] = df["sales"].astype(int)
    df["quantity"] = df["quantity"].astype(int)

    df_sales = df.drop(["id", "quantity"], axis=1)
    df_quantity = df.drop(["id", "sales"], axis=1)

    last_date = df["date"].max()
    date1 = datetime.strptime(last_date, "%Y-%m")
    date2 = datetime.strptime(to_date_param, "%Y-%m")

    difference = relativedelta.relativedelta(date2, date1)
    months_difference = difference.months + (12 * difference.years)

    df_pred_sales = next_month_sales_predict(df_sales, months_difference)
    df_pred_sales["pred_value"] = df_pred_sales["pred_value"].mask(
        df_pred_sales["pred_value"] == -1, df_pred_sales["sales"]
    )

    df_pred_quantity = next_month_quantity_predict(df_quantity, months_difference)
    df_pred_quantity["pred_value"] = df_pred_quantity["pred_value"].mask(
        df_pred_quantity["pred_value"] == -1, df_pred_quantity["quantity"]
    )

    merged_df = df_pred_quantity.merge(
        df_pred_sales, on="date", suffixes=("_quantity", "_sales")
    )

    for _, row in merged_df.iterrows():
        date = row["date"]
        if date > last_date_demand:
            sales = row["pred_value_sales"]
            quantity = row["pred_value_quantity"]
            # Demand.objects.delete(date=date)
            Demand.objects.create(date=date, sales=sales, quantity=quantity)

    filtered_demands = Demand.objects.filter(
        date__range=[from_date_param, to_date_param]
    )
    return DemandSerializer(filtered_demands, many=True).data


def predict_demand_by_months_from_date_with_model(from_date_param, to_date_param):
    queryset_demand = Demand.objects.all()

    df = pd.DataFrame.from_records(queryset_demand.values())

    df["sales"] = df["sales"].astype(int)
    df["quantity"] = df["quantity"].astype(int)

    df_sales = df.drop(["id", "quantity"], axis=1)
    df_quantity = df.drop(["id", "sales"], axis=1)

    last_date = df["date"].max()
    date1 = datetime.strptime(from_date_param, "%Y-%m")
    date2 = datetime.strptime(to_date_param, "%Y-%m")

    difference = relativedelta.relativedelta(date2, date1)
    months_difference = difference.months + (12 * difference.years)

    df_pred_sales = next_month_sales_predict(df_sales, months_difference)
    df_pred_sales["pred_value"] = df_pred_sales["pred_value"].mask(
        df_pred_sales["pred_value"] == -1, df_pred_sales["sales"]
    )

    df_pred_quantity = next_month_quantity_predict(df_quantity, months_difference)
    df_pred_quantity["pred_value"] = df_pred_quantity["pred_value"].mask(
        df_pred_quantity["pred_value"] == -1, df_pred_quantity["quantity"]
    )

    merged_df = df_pred_quantity.merge(
        df_pred_sales, on="date", suffixes=("_quantity", "_sales")
    )

    for _, row in merged_df.iterrows():
        date = row["date"]
        if date > last_date:
            sales = row["pred_value_sales"]
            quantity = row["pred_value_quantity"]
            Demand.objects.create(date=date, sales=sales, quantity=quantity)

    # return merged_df.to_dict(orient="records")
    filtered_demands = Demand.objects.filter(
        date__range=[from_date_param, to_date_param]
    )
    return DemandSerializer(filtered_demands, many=True).data


def predict_demand_by_months_from_date_to_date(from_date_param, to_date_param):
    filtered_demands = Demand.objects.filter(
        date__range=[from_date_param, to_date_param]
    )
    return DemandSerializer(filtered_demands, many=True).data


def next_month_sales_predict(df, quantity_months):
    df_final = pd.DataFrame()
    for month in range(quantity_months):
        last_month = str(list(df["date"])[-1])  # OBTENER EL ULTIMO MES (2021-12)
        format_date = datetime.strptime(
            last_month, "%Y-%m"
        )  # FORMATO DE PYTHON PARA LA FECHA
        relative_difference = relativedelta.relativedelta(
            months=1
        )  # OBTENES UN OBJETO RELATIVE PARA SUMAR UN MES A LA ULTIMA FECHA, ESTE RELATIVEDELTA ES PARA SUMAR FECHAS DE MANERA MAS PRECISA INCLUYENDO HASTA MINUTOS Y SEGUNDOS
        next_month = (format_date + relative_difference).strftime(
            "%Y-%m"
        )  # OBTENEMOS EL SIGUIENTE MES
        new_data = pd.DataFrame(data={"date": [next_month], "sales": [0]})
        df = pd.concat(
            [df, new_data], ignore_index=True
        )  # AGREGAMOS EL ULTIMO MES QUE SE OBTUVO Y SE LE PONE EL VALOR DE VENTAS EN 0, QUEDARIA POR EJEMPLO ASI: MES: 2022-01 y VENTAS: 0

        df_diff = df.copy()
        df_diff["prev_sales"] = df_diff["sales"].shift(1)
        df_diff = df_diff.dropna()
        df_diff["diff"] = df_diff["sales"] - df_diff["prev_sales"]
        df_supervised = df_diff.drop(["prev_sales"], axis=1)

        for inc in range(1, 13):
            field_name = "lag_" + str(inc)
            df_supervised[field_name] = df_supervised["diff"].shift(inc)

        df_supervised = df_supervised.dropna().reset_index(drop=True)
        df_model = df_supervised.drop(["sales", "date"], axis=1)
        train_set, test_set = df_model[:-1].values, df_model[-1:].values

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

        model = Sequential()
        model.add(
            LSTM(
                4,
                batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                stateful=True,
            )
        )
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(X_train, y_train, epochs=12, batch_size=1, verbose=1, shuffle=False)

        y_pred = model.predict(X_test, batch_size=1)
        y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

        pred_test_set = []
        for index in range(0, len(y_pred)):
            pred_test_set.append(np.concatenate([y_pred[index], X_test[index]], axis=1))

        pred_test_set = np.array(pred_test_set)
        pred_test_set = pred_test_set.reshape(
            pred_test_set.shape[0], pred_test_set.shape[2]
        )
        pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

        result_list = []
        sales_dates = list(df[-2:].date)
        act_sales = list(df[-2:].sales)

        result_dic = {}
        result_dic["pred_value"] = int(act_sales[0])
        result_dic["date"] = sales_dates[0]
        result_list.append(result_dic)

        for index in range(0, len(pred_test_set_inverted)):
            result_dict = {}
            result_dict["pred_value"] = int(
                pred_test_set_inverted[index][0] + act_sales[index]
            )
            result_dict["date"] = sales_dates[index + 1]
            result_list.append(result_dict)

        df_result = pd.DataFrame(result_list)
        if month == 0:
            df_final["pred_value"] = list(df_result["pred_value"])
            df_final["date"] = list(df_result["date"])
        else:
            df_final = df_final.append(
                pd.DataFrame(
                    data={
                        "pred_value": [list(df_result["pred_value"])[-1]],
                        "date": [list(df_result["date"])[-1]],
                    }
                )
            )

        df = df[:-1]
        df = df.append(
            pd.DataFrame(
                data={
                    "date": [next_month],
                    "sales": [list(df_result["pred_value"])[-1]],
                }
            )
        )
        df = df.reset_index().drop("index", axis=1)

    df_sales_pred = pd.merge(df, df_final, on="date", how="left")
    df_sales_pred = df_sales_pred.fillna(-1)
    return df_sales_pred


def next_month_quantity_predict(df, quantity_months):
    df_final = pd.DataFrame()
    for month in range(quantity_months):
        last_month = str(list(df["date"])[-1])  # OBTENER EL ULTIMO MES (2021-12)
        format_date = datetime.strptime(
            last_month, "%Y-%m"
        )  # FORMATO DE PYTHON PARA LA FECHA
        relative_difference = relativedelta.relativedelta(
            months=1
        )  # OBTENES UN OBJETO RELATIVE PARA SUMAR UN MES A LA ULTIMA FECHA, ESTE RELATIVEDELTA ES PARA SUMAR FECHAS DE MANERA MAS PRECISA INCLUYENDO HASTA MINUTOS Y SEGUNDOS
        next_month = (format_date + relative_difference).strftime(
            "%Y-%m"
        )  # OBTENEMOS EL SIGUIENTE MES
        new_data = pd.DataFrame(data={"date": [next_month], "quantity": [0]})
        df = pd.concat(
            [df, new_data], ignore_index=True
        )  # AGREGAMOS EL ULTIMO MES QUE SE OBTUVO Y SE LE PONE EL VALOR DE VENTAS EN 0, QUEDARIA POR EJEMPLO ASI: MES: 2022-01 y VENTAS: 0

        df_diff = df.copy()
        df_diff["prev_quantity"] = df_diff["quantity"].shift(1)
        df_diff = df_diff.dropna()
        df_diff["diff"] = df_diff["quantity"] - df_diff["prev_quantity"]
        df_supervised = df_diff.drop(["prev_quantity"], axis=1)

        for inc in range(1, 13):
            field_name = "lag_" + str(inc)
            df_supervised[field_name] = df_supervised["diff"].shift(inc)

        df_supervised = df_supervised.dropna().reset_index(drop=True)
        df_model = df_supervised.drop(["quantity", "date"], axis=1)
        train_set, test_set = df_model[:-1].values, df_model[-1:].values

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

        model = Sequential()
        model.add(
            LSTM(
                4,
                batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                stateful=True,
            )
        )
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(X_train, y_train, epochs=12, batch_size=1, verbose=1, shuffle=False)

        y_pred = model.predict(X_test, batch_size=1)
        y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])

        pred_test_set = []
        for index in range(0, len(y_pred)):
            pred_test_set.append(np.concatenate([y_pred[index], X_test[index]], axis=1))

        pred_test_set = np.array(pred_test_set)
        pred_test_set = pred_test_set.reshape(
            pred_test_set.shape[0], pred_test_set.shape[2]
        )
        pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

        result_list = []
        sales_dates = list(df[-2:].date)
        act_quantity = list(df[-2:].quantity)

        result_dic = {}
        result_dic["pred_value"] = int(act_quantity[0])
        result_dic["date"] = sales_dates[0]
        result_list.append(result_dic)

        for index in range(0, len(pred_test_set_inverted)):
            result_dict = {}
            result_dict["pred_value"] = int(
                pred_test_set_inverted[index][0] + act_quantity[index]
            )
            result_dict["date"] = sales_dates[index + 1]
            result_list.append(result_dict)

        df_result = pd.DataFrame(result_list)
        if month == 0:
            df_final["pred_value"] = list(df_result["pred_value"])
            df_final["date"] = list(df_result["date"])
        else:
            df_final = df_final.append(
                pd.DataFrame(
                    data={
                        "pred_value": [list(df_result["pred_value"])[-1]],
                        "date": [list(df_result["date"])[-1]],
                    }
                )
            )

        df = df[:-1]
        df = df.append(
            pd.DataFrame(
                data={
                    "date": [next_month],
                    "quantity": [list(df_result["pred_value"])[-1]],
                }
            )
        )
        df = df.reset_index().drop("index", axis=1)

    df_sales_pred = pd.merge(df, df_final, on="date", how="left")
    df_sales_pred = df_sales_pred.fillna(-1)
    return df_sales_pred
