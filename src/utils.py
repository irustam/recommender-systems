from config import pseudo_item_id


def prefilter_items(data, take_n_popular=5000, popular_col='quantity', min_value=1, max_value=30, week_left=52):
    """Предфильтрация товаров"""
    data2 = data.copy()
    
    # 1. Удаление товаров, со средней ценой < 1$
#     data2.loc[data2['sales_value']/data2['quantity'] < min_value, 'item_id'] = pseudo_item_id
#     data2 = data2.loc[data2['sales_value']/data2['quantity'] > min_value]
    
    # 2. Удаление товаров со соедней ценой > 30$
#     data2.loc[data2['sales_value']/data2['quantity'] > max_value, 'item_id'] = pseudo_item_id
#     data2 = data2.loc[data2['sales_value']/data2['quantity'] < max_value]
    
    # 3. Придумайте свой фильтр
    # 3.1 Убираем непонятную положительную скидку
#     data2.loc[data2['retail_disc'] > 0, 'item_id'] = pseudo_item_id
#     data2 = data2.loc[data2['retail_disc'] <= 0]
    
    # 3.2 Оставляем только товары, купленные в последние N недель:
    # from_week = data2['week_no'].max() - week_left
    # data2 = data2.loc[data2['week_no'] > from_week]
    
    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    popularity = data2.groupby('item_id')[popular_col].sum().reset_index()
    top_items = popularity.sort_values(popular_col, ascending=False).head(take_n_popular).item_id.tolist()
    data2.loc[~data2['item_id'].isin(top_items), 'item_id'] = pseudo_item_id

    
    return data2


def get_ctm_df(data_features):
    """data_features - датасет с фичами товаров"""
    items_brand = data_features[['item_id', 'brand']]
    ctm_map = {'National': 0, 'Private': 1}
    items_brand['brand'] = items_brand['brand'].map(ctm_map)
    items_brand.rename(columns={'brand': 'ctm'}, inplace=True)
    
    return items_brand.set_index('item_id').to_dict()['ctm']