import pandas as pd
import numpy as np
from config import pseudo_item_id

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, ctm, top_col='quantity', weighting=True):
        """
        data - датафрейм с покупками юзер - товар
        ctm - словарь принадлежности товара к СТМ {item_id: 0/1}
        top_col - фича, по которой будем определять популярность товаров
        """
                
        # Топ покупок каждого юзера
        self.top_col = top_col
        self.top_purchases = data.groupby(['user_id', 'item_id'])[self.top_col].count().reset_index()
        self.top_purchases.sort_values(self.top_col, ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != pseudo_item_id]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')[self.top_col].count().reset_index()
        self.overall_top_purchases.sort_values(self.top_col, ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != pseudo_item_id]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.user_item_matrix = self.prepare_matrix(data, top_col)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = ctm
        
        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data, top_col):
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

        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
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

    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        # Не забывайте, что нужно учесть параметр filter_ctm
        sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()
        filter_items = [self.itemid_to_id[pseudo_item_id]]
        # filter_items = None
        
        if filter_ctm:
            filter_items.expand([k for k, v in self.item_id_to_ctm.items() if v==1])
        
        res = [self.id_to_itemid[rec[0]] for rec in 
                    self.own_recommender.recommend(userid=self.userid_to_id[user], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=True,
                                    filter_items=filter_items,
                                    recalculate_user=True)]

        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        sparse_user_item = csr_matrix(self.user_item_matrix).tocsr()
        filter_items = [self.itemid_to_id[pseudo_item_id]]
        
        res = [self.id_to_itemid[rec[0]] for rec in 
                    self.model.recommend(userid=self.userid_to_id[user], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=filter_items, 
                                    recalculate_user=True)]

        return res