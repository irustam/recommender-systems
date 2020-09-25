import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from config import pseudo_item_id


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, ctm=None, top_col='quantity', weighting=True):
        """
        data - датафрейм с покупками юзер - товар
        ctm - словарь принадлежности товара к СТМ {item_id: 0/1}
        top_col - фича, по которой будем определять популярность товаров
        """
                
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])[top_col].count().reset_index()
        self.top_purchases.sort_values(top_col, ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != pseudo_item_id]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')[top_col].count().reset_index()
        self.overall_top_purchases.sort_values(top_col, ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != pseudo_item_id]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.user_item_matrix = self.__prepare_matrix(data, top_col)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.__prepare_dicts(self.user_item_matrix)
        
        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = ctm
        
        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T
            print('weighting done')
        
        self.model = self.fit(self.user_item_matrix)

    @staticmethod
    def __prepare_matrix(data, top_col):
        if top_col == 'sales_value':
            aggfunc = 'sum'
        else:
            aggfunc = 'count'
            
            
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values=top_col, # Можно пробовать другие варианты
                                  aggfunc=aggfunc, 
                                  fill_value=0
                                 )
        
        user_item_matrix[user_item_matrix > 0] = 1 # так как в итоге хотим предсказать 
        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix

    @staticmethod
    def __prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)
        
        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors,
                                         regularization=regularization,
                                         iterations=iterations,
                                         num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)
        
        return model

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""
        # print('start recomendation')
        res = []
        if user in self.userid_to_id.keys():
            res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                            user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                            N=N,
                                            filter_already_liked_items=False,
                                            filter_items=[self.itemid_to_id[999999]],
                                            recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        res = []
        if user in self.userid_to_id.keys():
            # Находим топ-N похожих пользователей
            similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
            similar_users = [rec[0] for rec in similar_users]
            similar_users = similar_users[1:]   # удалим юзера из запроса

            for user in similar_users:
                res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res