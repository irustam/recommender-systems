from config import pseudo_item_id
from metrics import precision_at_k, money_precision_at_k
import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000, item_features=None):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        # создаем датафрейм с кол-вом уникальных товаров, сгруппированных по категориям
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()
        
        department_size.columns = ['department', 'n_items']
        
        # выделяем категории, где менее 150 уникальных товаров в отдельный список
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        
        # получаем список товаров, которые представлены в категориях, где менее 150 уникальных товаров
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()
        
        # убираем из датасета data список товаров, полученных выше items_in_rare_departments
        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    # ...

    return data


def get_ctm(data_features):
    """data_features - датасет с фичами товаров"""
    # items_brand = data_features[['item_id', 'brand']]
    # ctm_map = {'National': 0, 'Private': 1}
    # items_brand['brand'] = items_brand['brand'].map(ctm_map)
    # items_brand.rename(columns={'brand': 'ctm'}, inplace=True)
    #
    # return items_brand.set_index('item_id').to_dict()['ctm']

    return data_features.loc[data_features['brand'] == 'Private', 'item_id'].to_list()


class Result:
    def __init__(self, data):
        self.__result = data.groupby('user_id')['item_id'].unique().reset_index()
        self.__result.columns = ['user_id', 'actual']
        self.__precision_at_k = {}

    def add_result(self, rec_model_column_name, rec_model, N=5, filter_ctm=False):
        self.__result[rec_model_column_name] = self.__result['user_id'].apply(
            lambda x: rec_model(x, filter_ctm=filter_ctm, N=N))

        return True

    def count_presision(self, rec_model_column_name):
        self.__precision_at_k['p_at_k_' + rec_model_column_name] = self.__result.apply(
            lambda row: precision_at_k(row[rec_model_column_name], row['actual']), axis=1).mean()

        return True

    def get_result(self):
        return self.__result

    def get_precision_at_k(self, rec_model_column_name):
        return self.__precision_at_k['p_at_k_' + rec_model_column_name]


# def prefilter_items(data, take_n_popular=5000, popular_col='quantity', min_value=1, max_value=30):
#     """Предфильтрация товаров"""
#     data2 = data.copy()
#
#     # Уберем самые популярные товары (их и так купят)
#     popularity = data2.groupby('item_id')['user_id'].nunique().reset_index() / data2['user_id'].nunique()
#     popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
#
#     top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
#     data2 = data2[~data2['item_id'].isin(top_popular)]
#
#     # Уберем самые НЕ популярные товары (их и так НЕ купят)
#     top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
#     data2 = data2[~data2['item_id'].isin(top_notpopular)]
#
#     # Уберем товары, которые не продавались за последние 12 месяцев
#
#     # Уберем не интересные для рекоммендаций категории (department)
#     if item_features is not None:
#         department_size = pd.DataFrame(item_features.\
#                                         groupby('department')['item_id'].nunique().\
#                                         sort_values(ascending=False)).reset_index()
#
#         department_size.columns = ['department', 'n_items']
#         rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
#         items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()
#
#         data2 = data2[~data2['item_id'].isin(items_in_rare_departments)]
#
#
#     # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
#     data2['price'] = data2['sales_value'] / (np.maximum(data2['quantity'], 1))
#     data2 = data2[data2['price'] > min_value]
#
#     # Уберем слишком дорогие товарыs
#     data2 = data2[data2['price'] < max_value]
#
#     # Возбмем топ по популярности
#     popularity = data2.groupby('item_id')[popular_col].sum().reset_index()
#     popularity.rename(columns={popular_col: 'n_sold'}, inplace=True)
#
#     top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
#
#     # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
#     data2.loc[~data2['item_id'].isin(top), 'item_id'] = pseudo_item_id
#
#     return data2

