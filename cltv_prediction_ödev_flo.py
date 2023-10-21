##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
pip install lifetimes
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

###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("/Users/mrpurtas/Desktop/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
"""Elbette, notta belirtilen bilgiyi açıklayayım:

CLTV (Müşteri Ömür Boyu Değeri) hesaplamalarında frekans değerlerinin tamsayı olması gerekmektedir.
Yani bir müşterinin belirli bir süre zarfında kaç kez satın alma yaptığı bilgisini temsil eden frekans, kesirli bir 
değer olmamalıdır. Örneğin, bir müşteri 2.5 kez satın alma yapmış olamaz. Bu nedenle frekans değerleri tam sayı olmalıdır.

Ancak aykırı değer baskılama işlemlerinde bazen, belirlediğiniz üst ve alt sınırlar kesirli değerler olabilir. 
Örneğin, bir üst sınır değeri 2.78 olarak hesaplandığında ve bu değeri frekansta kullanmak istediğinizde bu 
değeri yuvarlamalısınız. Bu yuvarlama işlemi, round() fonksiyonu ile gerçekleştirilir.

round() fonksiyonu, verilen bir sayıyı en yakın tam sayıya yuvarlar. Örneğin:

round(2.1) sonucu 2 olur.
round(2.8) sonucu 3 olur.
Bu nedenle, outlier_thresholds fonksiyonundan dönen alt ve üst sınırları kullanmadan önce frekans değerlerini 
yuvarlamak için round() fonksiyonunu kullanmalısınız. Bu, kesirli değerleri tam sayıya dönüştürerek frekansın doğru 
bir şekilde tamsayı olarak temsil edilmesini sağlar.

"""

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.
df.info()
df.describe().T
df.describe([0.01,0.25,0.5,0.75,0.99]).T

df.head()
df.isnull().sum()
df.dropna(inplace=True)
outlier_thresholds(df, "order_num_total_ever_online")
outlier_thresholds(df, "order_num_total_ever_offline")
outlier_thresholds(df, "customer_value_total_ever_offline")
outlier_thresholds(df, "customer_value_total_ever_online")

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

threshold_columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
                     "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in threshold_columns:
    replace_with_thresholds(df, col)


# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_order_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_list = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
for col in date_list:
    df[col] = pd.to_datetime(df[col])

df.dtypes

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]


cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] -
                                   df["first_order_date"]).astype('timedelta64[D]')) / 7

cltv_df["T_weekly"] = ((today_date -
                        df["first_order_date"]).astype('timedelta64[D]')) / 7


cltv_df["frequency"] = df["total_order_num"].astype(int)
cltv_df["monetary_cltv_avg"] = df["total_order_value"] / df["total_order_num"]
cltv_df[cltv_df["frequency"] > 1]

# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.


# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'], cltv_df["T_weekly"])
# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

plot_period_transactions(bgf)
plt.show()
# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values(by="cltv", ascending=False).head(20)

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.
# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], q=4, labels=["D", "C", "B", "A"])

cltv_df.sort_values(by="cltv", ascending=False, inplace=True)
cltv_df

# BONUS: Tüm süreci fonksiyonlaştırınız.
def create_cltv_df(dataframe):
    df = dataframe.copy()

    def outlier_thresholds(df, variable):
        quartile1 = df[variable].quantile(0.01)
        quartile3 = df[variable].quantile(0.99)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(df, variable):
        low_limit, up_limit = outlier_thresholds(df, variable)
        df.loc[(df[variable] < low_limit), variable] = round(low_limit)
        df.loc[(df[variable] > up_limit), variable] = round(up_limit)

    threshold_columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
                         "customer_value_total_ever_offline", "customer_value_total_ever_online"]

    for col in threshold_columns:
        replace_with_thresholds(df, col)

    df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["total_order_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    date_list = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    for col in date_list:
        df[col] = pd.to_datetime(df[col])
    today_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = df["master_id"]

    cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] -
                                       df["first_order_date"]).astype('timedelta64[D]')) / 7

    cltv_df["T_weekly"] = ((today_date -
                            df["first_order_date"]).astype('timedelta64[D]')) / 7

    cltv_df["frequency"] = df["total_order_num"].astype(int)
    cltv_df["monetary_cltv_avg"] = df["total_order_value"] / df["total_order_num"]
    cltv_df[cltv_df["frequency"] > 1]
    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'], cltv_df["T_weekly"])
    cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3, cltv_df["frequency"],
                                                                                           cltv_df[
                                                                                               "recency_cltv_weekly"],
                                                                                           cltv_df["T_weekly"])
    cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6, cltv_df["frequency"],
                                                                                           cltv_df[
                                                                                               "recency_cltv_weekly"],
                                                                                           cltv_df["T_weekly"])
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])
    cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency_cltv_weekly'],
                                                  cltv_df['T_weekly'],
                                                  cltv_df['monetary_cltv_avg'],
                                                  time=6,
                                                  freq="W",
                                                  discount_rate=0.01)
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], q=4, labels=["D", "C", "B", "A"])

    cltv_df.sort_values(by="cltv", ascending=False, inplace=True)
    return cltv_df


dataframe = pd.read_csv("/Users/mrpurtas/Desktop/FLOCLTVPrediction/flo_data_20k.csv")
create_cltv_df(dataframe)


###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################


# 1. OmniChannel.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("/Users/mrpurtas/Desktop/FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.
df.describe().T

threshold_columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
                         "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in threshold_columns:
    replace_with_thresholds(df, col)


# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_order_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_list = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]

for col in date_list:
    df[col] = pd.to_datetime(df[col])

df.dtypes

###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = (((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7)
cltv_df["T_weekly"] = (((today_date - df["first_order_date"]).astype('timedelta64[D]')) / 7)
cltv_df["frequency"] = df["total_order_num"].astype(int)
cltv_df["monetary_cltv_avg"] = df["total_order_value"] / df["total_order_num"]

###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'], cltv_df["T_weekly"])


# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(24, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])


# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz.

cltv_df.sort_values(by="exp_sales_3_month", ascending=False).head(10)
cltv_df.sort_values(by="exp_sales_6_month", ascending=False).head(10)

# 2.  Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df["monetary_cltv_avg"])



# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                            cltv_df["frequency"],
                            cltv_df["recency_cltv_weekly"],
                            cltv_df["T_weekly"],
                            cltv_df["monetary_cltv_avg"],
                            time=6, freq="W", discount_rate=0.01)
# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values(by="cltv", ascending=False)


###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], q=4, labels=["D", "C", "B", "A"])

# 2. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
cltv_df.groupby("cltv_segment").agg({"recency_cltv_weekly": "mean", "frequency": "mean", "monetary_cltv_avg": "mean"})







