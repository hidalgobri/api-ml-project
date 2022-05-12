import numpy as np
import pickle
import pandas as pd
from pytest import console_main
from sklearn.decomposition import PCA
from pandas import DataFrame
from flask import Flask, request, render_template, flash
# from sqlalchemy import true # Import flask libraries
from starlette.responses import HTMLResponse
import tensorflow as tf
import re
import sys

app = Flask(__name__, template_folder="templates")
app.secret_key = "hovabrihica11"


def generarPCA(n_componentes, X):
    pca = PCA(n_components=n_componentes)
    cp = pca.fit_transform(X)
    return pd.DataFrame(data=cp)


def isNumber(element):
    return False if re.match(r'^-?\d+(\.?\d+)?$', element) is None else True


def isPositiveInt(element):
    return False if re.match(r'^\d+$', element) is None else True


def formValidation(price, rating_product, retail_price, rating_count_product,
                   product_inventory, shipping_price, countries_shipped_to,
                   inventory_total, merchant_rating, merchant_rating_count

                   ):
    return True if (price and rating_product and retail_price and rating_count_product
                    and product_inventory and shipping_price and countries_shipped_to 
                    and inventory_total and merchant_rating and merchant_rating_count) else False


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def classify_type():
    try:
        # Jalar datos del form

        price = float(request.form['price']) if isNumber(
            request.form['price']) else None  # Get parameters

        retail_price = int(request.form['retail_price']) if isNumber(
            request.form['retail_price']) else None

        rating_product = float(request.form['rating_product']) if isNumber(
            request.form['rating_product']) else None

        color_product = request.form['color_product']

        rating_count_product = int(request.form['rating_count_product']) if isPositiveInt(
            request.form['rating_count_product']) else None

        talla_product = request.form['talla_product']

        product_inventory = int(request.form['product_inventory']) if isPositiveInt(
            request.form['product_inventory']) else None

        shipping_price = int(request.form['shipping_price']) if isNumber(
            request.form['shipping_price']) else None

        countries_shipped_to = int(request.form['countries_shipped_to']) if isPositiveInt(
            request.form['countries_shipped_to']) else None

        inventory_total = int(request.form['inventory_total']) if isPositiveInt(
            request.form['inventory_total']) else None

        merchant_info_subtitle = request.form['merchant_info_subtitle']

        merchant_rating = float(request.form['merchant_rating']) if isNumber(
            request.form['merchant_rating']) else None

        merchant_rating_count = int(request.form['merchant_rating_count']) if isPositiveInt(
            request.form['merchant_rating_count']) else None

        product_color_black = 0
        product_color_blue = 0
        product_color_brown = 0
        product_color_green = 0
        product_color_grey = 0
        product_color_orange = 0
        product_color_pink = 0
        product_color_purple = 0
        product_color_red = 0
        product_color_stamping = 0
        product_color_white = 0
        product_color_yellow = 0

        product_variation_size_id_l = 0
        product_variation_size_id_m = 0
        product_variation_size_id_no_size = 0
        product_variation_size_id_s = 0
        product_variation_size_id_xl = 0
        product_variation_size_id_xs = 0

        shipping_option_price_1 = 0
        shipping_option_price_2 = 0
        shipping_option_price_3 = 0
        shipping_option_price_4 = 0
        shipping_option_price_5 = 0
        shipping_option_price_6 = 0
        shipping_option_price_7 = 0
        shipping_option_price_12 = 0

        merchant_info_subtitle_bueno = 0
        merchant_info_subtitle_excelente = 0
        merchant_info_subtitle_malo = 0
        merchant_info_subtitle_otros = 0
        merchant_info_subtitle_regular = 0

        if color_product.casefold() == 'amarillo':
            product_color_yellow = 1
        elif color_product.casefold() == 'azul':
            product_color_blue = 1
        elif color_product.casefold() == 'blanco':
            product_color_white = 1
        elif color_product.casefold() == 'cafe':
            product_color_brown = 1
        elif color_product.casefold() == 'estampado':
            product_color_stamping = 1
        elif color_product.casefold() == 'naranja':
            product_color_orange = 1
        elif color_product.casefold() == 'negro':
            product_color_black = 1
        elif color_product.casefold() == 'plomo':
            product_color_grey = 1
        elif color_product.casefold() == 'rojo':
            product_color_red = 1
        elif color_product.casefold() == 'rosa':
            product_color_pink = 1
        elif color_product.casefold() == 'verde':
            product_color_green = 1
        elif color_product.casefold() == 'morado':
            product_color_purple = 1

        if talla_product.casefold() == 's':
            product_variation_size_id_s = 1
        elif talla_product.casefold() == 'xs':
            product_variation_size_id_xs = 1
        elif talla_product.casefold() == 'm':
            product_variation_size_id_m = 1
        elif talla_product.casefold() == 'l':
            product_variation_size_id_l = 1
        elif talla_product.casefold() == 'xl':
            product_variation_size_id_xl = 1
        elif talla_product.casefold() == 'no_size':
            product_variation_size_id_no_size = 1

        if shipping_price == '1':
            shipping_option_price_1 = 1
        elif shipping_price == '2':
            shipping_option_price_2 = 1
        elif shipping_price == '3':
            shipping_option_price_3 = 1
        elif shipping_price == '4':
            shipping_option_price_4 = 1
        elif shipping_price == '5':
            shipping_option_price_5 = 1
        elif shipping_price == '6':
            shipping_option_price_6 = 1
        elif shipping_price == '7':
            shipping_option_price_7 = 1
        elif shipping_price == '12':
            shipping_option_price_12 = 1

        if merchant_info_subtitle.casefold() == 'excelentes':
            merchant_info_subtitle_excelente = 1
        elif merchant_info_subtitle.casefold() == 'buenos':
            merchant_info_subtitle_bueno = 1
        elif merchant_info_subtitle.casefold() == 'regular':
            merchant_info_subtitle_regular = 1
        elif merchant_info_subtitle.casefold() == 'malos':
            merchant_info_subtitle_malo = 1
        elif merchant_info_subtitle.casefold() == 'otros':
            merchant_info_subtitle_otros = 1

        # creo una lista de esos datos
        if formValidation(price, rating_product, retail_price, rating_count_product,
                          product_inventory, shipping_price, countries_shipped_to, inventory_total,
                          merchant_rating, merchant_rating_count):
            data = [[price, retail_price, rating_product, rating_count_product,
                     product_inventory,
                         countries_shipped_to, inventory_total,
                     merchant_rating_count,
                         merchant_rating, product_color_black,
                     product_color_blue, product_color_brown, product_color_green, product_color_grey,
                     product_color_orange, product_color_pink, product_color_purple, product_color_red,
                     product_color_stamping, product_color_white, product_color_yellow, product_variation_size_id_l,
                     product_variation_size_id_m, product_variation_size_id_no_size, product_variation_size_id_s,
                     product_variation_size_id_xl, product_variation_size_id_xs,
                     shipping_option_price_1, shipping_option_price_2, shipping_option_price_3, shipping_option_price_4,
                     shipping_option_price_5, shipping_option_price_6, shipping_option_price_7, shipping_option_price_12,
                     merchant_info_subtitle_bueno, merchant_info_subtitle_excelente, merchant_info_subtitle_malo, merchant_info_subtitle_otros,
                     merchant_info_subtitle_regular
                     ]]

            # cabeceras para el dataframe
            cabeceras_df = ['price', 'retail_price', 'rating',
                            'rating_count', 'product_variation_inventory', 'countries_shipped_to', 'inventory_total',
                            'merchant_rating_count', 'merchant_rating']

            cabeceras_color = ['product_color_black',
                               'product_color_blue', 'product_color_brown', 'product_color_green', 'product_color_grey',
                               'product_color_orange', 'product_color_pink', 'product_color_purple', 'product_color_red',
                               'product_color_stamping', 'product_color_white', 'product_color_yellow']
            cabeceras_talla_product = ['product_variation_size_id_l',
                                       'product_variation_size_id_m', 'product_variation_size_id_no_size', 'product_variation_size_id_s',
                                       'product_variation_size_id_xl', 'product_variation_size_id_xs']
            cabeceras_shipping = ['shipping_option_price_1', 'shipping_option_price_2', 'shipping_option_price_3', 'shipping_option_price_4',
                                  'shipping_option_price_5', 'shipping_option_price_6', 'shipping_option_price_7', 'shipping_option_price_12']
            cabeceras_subtitle = ['merchant_info_subtitle_bueno', 'merchant_info_subtitle_excelente', 'merchant_info_subtitle_malo', 'merchant_info_subtitle_otros',
                                  'merchant_info_subtitle_regular']

            for i in cabeceras_color:
                cabeceras_df.append(i)
            for j in cabeceras_talla_product:
                cabeceras_df.append(j)
            for k in cabeceras_shipping:
                cabeceras_df.append(k)
            for l in cabeceras_subtitle:
                cabeceras_df.append(l)

            # hago un DataFrame con esos datos y con esas cabeceras
            df = pd.DataFrame(data, columns=cabeceras_df)

            # transformo el tipo de dato de ciertas columnas del dataframe
            for i in cabeceras_color:
                df[i] = df[i].astype(np.uint8)
            for j in cabeceras_talla_product:
                df[j] = df[j].astype(np.uint8)
            for k in cabeceras_shipping:
                df[k] = df[k].astype(np.uint8)
            for l in cabeceras_subtitle:
                df[l] = df[l].astype(np.uint8)

            # cargo el modelo
            modelo_cargado = pickle.load(open('modelo_svr_sin_pca.sav', 'rb'))
            resultado_float = modelo_cargado.predict(df)
            resultado_str = 'Total de unidades a vender: ' + \
                resultado_float[0].astype('str')
            # devuelve el resultado
            return render_template('output.html', variety=resultado_str)
        else:
            flash("Entrada invalida de datos.")
            return render_template('home.html')
    except:
        flash("Algo ha salido mal, no se pudo procesar tu solicitud. Intenta nuevamente")
        return render_template('home.html')


# Run the Flask server
if(__name__ == '__main__'):
    app.run(debug=True)
