import pandas as pd
import datetime as dt
! pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

#1.1
data = pd.read_csv("C:/Users/nejat/OneDrive/Desktop/FLO_CLTV_Tahmini/flo_data_20k.csv")

# df_ = pd.read_csv(r"C:\Users\furka\Desktop\VBO DSMLBC-6\DSMLBC-8\Hafta 2\Modül_2_CRM_Analitigi\Dataset\flo_data_20K.csv")

df= data.copy()

df.describe().T

# 1.2 Aykırıdeğerleribaskılamakiçingerekliolanoutlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayalim
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt veüstlimitlerini round() ile yuvarlayacagiz.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable]<low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable]>up_limit), variable] = round(up_limit, 0)

# virgülden sonraki variable ne işe yarıyor ?

df[df["order_num_total_ever_online"] > 48] #pandas dataframe
df[df["order_num_total_ever_online"] > 48]['order_num_total_ever_online'] # pandas series
df.loc[df["order_num_total_ever_online"] > 48, "order_num_total_ever_online"] #pandas series


# 1.3 order_num_total_ever_online , order_num_total_ever_offline, customer_value_total_ever_offline
# customer_value_total_ever_online degerlerini baskilama.

df["order_num_total_ever_online"].max() #200
df["order_num_total_ever_offline"].max() #109


columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df, col)


df["order_num_total_ever_online"].max() #48
df["order_num_total_ever_offline"].max() #16


# verimin ilk hali df_ den gözlemleyelim.
df_[df_["order_num_total_ever_online"] > 48]["order_num_total_ever_online"].count() #bu kadar outlierım varmış
df[df["order_num_total_ever_online"] == 48]["order_num_total_ever_online"].count() #yani bu kadar değeri baskılamışım


# 1.4 Omnichannel müşterilerin hem online'dan hem de offline platformlar dan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturalim

df['Total_number_of_orders'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

df['Total_Expenditure'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

# 1.5 Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e cevirelim.

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# 2 Veri setindeki enson alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alıyoruz.


df['last_order_date'].max()

analysis_date = dt.datetime(2021,6,2)

# Customer_id, recency_cltv_weekly, T_cltv_weekly, frequency, monetary_cltv_avg degerlelrini tanimlayalim.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["Total_number_of_orders"]
cltv_df["monetary_cltv_avg"] = df["Total_Expenditure"] / df["Total_number_of_orders"]

cltv_df.head()


#3.1 3 ay içerisinde müşteriler'den beklenen satın almaları tahmin ediniz ve
# exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
#6 ay içerisinde müşterilerden beklenen satınalmaları tahminedinizve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.


bgf = BetaGeoFitter(penalizer_coef = 0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

cltv_df["exp_sales_3_month"] = bgf.predict(4*3, #3 ay oldugu icin 4*3 yazdik.
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])
cltv_df.head(10)
cltv_df["exp_sales_3_month"].max() #4.65
cltv_df.loc[cltv_df["exp_sales_3_month"] == 4.644667400371308, "customer_id"] #3ay içerisinde predict edilen en yüksek sales kime?
cltv_df.loc[cltv_df["exp_sales_3_month"] == 4.644667400371308] #52 adet alışveriş


# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyelim.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6, #6 ay tahmin etcegimiz icin 4*6 yazdik.
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])
cltv_df.head()

# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyelim. Fark var mı? yok

cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]


# 2.  Gamma-Gamma modelini fit edelim. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyelim.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()


# 3. 6 aylık CLTV tahmini hesaplayalim ve cltv ismiyle dataframe'e ekleyelim.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

cltv_df.head()

# CLTV değeri en yüksek 20 kişiyi gözlemleyelim.

cltv_df.sort_values("cltv",ascending=False)[:20]

cltv_df["cltv"].sum() # şirketin 6 ay sonraki toplam beklenen değeri

###############################################################
# 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 4.1 6 aylık standartlaştırılmış CLTV'ye göre tüm müşterilerimizi 4 gruba (segmente) ayıralim ve grup isimlerini veri setine ekleyelim
# cltv_segment ismi ile atayalim

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# 4.2 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulunalim
cltv_df.groupby("cltv_segment").agg({"frequency":("mean","std"),
                                     "monetary_cltv_avg": "std"})


cltv_df.groupby("cltv_segment").agg(["mean","std","median"])







