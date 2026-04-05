import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def build_features(df):
    df = df.copy()
    # Target
    df['TotalNetProfit'] = (df['InStoreNetProfit'] + df['UberEatsNetProfit'] +
                            df['DoorDashNetProfit'] + df['SelfDeliveryNetProfit'])
    df['NetProfitPerOrder'] = df['TotalNetProfit'] / df['MonthlyOrders']
    # Revenue ratios
    total_rev = (df['InStoreRevenue'] + df['UberEatsRevenue'] +
                 df['DoorDashRevenue'] + df['SelfDeliveryRevenue'])
    df['InStoreRevShare']  = df['InStoreRevenue'] / total_rev
    df['UERevShare']       = df['UberEatsRevenue'] / total_rev
    df['DDRevShare']       = df['DoorDashRevenue'] / total_rev
    df['SDRevShare']       = df['SelfDeliveryRevenue'] / total_rev
    # Cost-to-revenue ratios
    df['COGS_Rev_Ratio']   = df['COGSRate']
    df['OPEX_Rev_Ratio']   = df['OPEXRate']
    df['TotalCostRate']    = df['COGSRate'] + df['OPEXRate']
    # Interaction terms
    df['Commission_UE']    = df['CommissionRate'] * df['UE_share']
    df['Commission_DD']    = df['CommissionRate'] * df['DD_share']
    df['DeliveryCost_SD']  = df['DeliveryCostPerOrder'] * df['SD_share']
    # Growth-adjusted demand
    df['GrowthAdjOrders']  = df['MonthlyOrders'] * df['GrowthFactor']
    df['GrowthAdjRevenue'] = total_rev * df['GrowthFactor']
    # Profit per order by channel
    df['InStorePPO']  = df['InStoreNetProfit']  / (df['InStoreOrders']  + 1)
    df['UE_PPO']      = df['UberEatsNetProfit']  / (df['UberEatsOrders']  + 1)
    df['DD_PPO']      = df['DoorDashNetProfit']  / (df['DoorDashOrders']  + 1)
    df['SD_PPO']      = df['SelfDeliveryNetProfit'] / (df['SelfDeliveryOrders'] + 1)
    # Encode categoricals
    for col in ['CuisineType', 'Segment', 'Subregion']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        joblib.dump(le, f'models/le_{col}.pkl')
    return df

FEATURE_COLS = [
    'AOV', 'MonthlyOrders', 'GrowthFactor',
    'COGSRate', 'OPEXRate', 'CommissionRate',
    'DeliveryRadiusKM', 'DeliveryCostPerOrder',
    'InStoreShare', 'UE_share', 'DD_share', 'SD_share',
    'InStoreRevShare', 'UERevShare', 'DDRevShare', 'SDRevShare',
    'TotalCostRate', 'Commission_UE', 'Commission_DD', 'DeliveryCost_SD',
    'GrowthAdjOrders', 'GrowthAdjRevenue',
    'InStorePPO', 'UE_PPO', 'DD_PPO', 'SD_PPO',
    'CuisineType_enc', 'Segment_enc', 'Subregion_enc'
]

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    file_path = r'C:\Users\ASUS\Documents\skycity_project\data\SkyCity Auckland Restaurants & Bars.csv'

    df = pd.read_csv(file_path)
    print("File loaded successfully! Shape:", df.shape)

    df = build_features(df)
    df.to_csv('data/features.csv', index=False)
    print("Features saved. Shape:", df.shape)
    print("Target stats:\n", df['TotalNetProfit'].describe())