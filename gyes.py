from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import joblib

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/board')
def board():
       return render_template('board.html')

@app.route('/mtpd')
def material():
    return render_template('mtpd.html')

@app.route('/algo', methods=['GET', 'POST'])
def aglo():
    if request.method == 'GET':
        return render_template('algo.html')

    if request.method == 'POST':
        
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, [None, 42])
        Y = tf.placeholder(tf.float32, [None, 1])
        W1 = tf.Variable(tf.random.normal([42, 8]))
        b1 = tf.Variable(tf.random.normal([8]))
        W2 = tf.Variable(tf.random.normal([8, 1]))
        b2 = tf.Variable(tf.random.normal([1]))
        hypothesis = tf.add(b2, tf.matmul(tf.add(b1, tf.matmul(X, W1)), W2))

        inpdata = pd.DataFrame({
        'PP_rate1': [0.], 'PP_FM1': [0.], 'PP_MI1': [0.],
        'PP_rate2': [0.], 'PP_FM2': [0.], 'PP_MI2': [0.],
        'PP_rate3': [0.], 'PP_FM3': [0.], 'PP_MI3': [0.],
        'PP_rate4': [0.], 'PP_FM4': [0.], 'PP_MI4': [0.],
        'R_rate1': [0.], 'R_density1': [0.], 'R_TS1': [0.],
        'R_rate2': [0.], 'R_density2': [0.], 'R_TS2': [0.],
        'R_rate3': [0.], 'R_density3': [0.], 'R_TS3': [0.],
        'R_rate4': [0.], 'R_density4': [0.], 'R_TS4': [0.],
        'F001': [0.], 'F002': [0.], 'F003': [0.],
        'F004': [0.], 'F005': [0.], 'F006': [0.],
        'F007': [0.], 'F008': [0.], 'F009': [0.],
        'F010': [0.], 'F011': [0.], 'F012': [0.],
        'F013': [0.], 'F014': [0.], 'F015': [0.],
        'F016': [0.], 'F017': [0.], 'F018': [0.],
         })


        outdata = [0, 0, 0, 0, 0, 0, 0]
        ppvalue = pd.read_excel('./model/PPvalue.xlsx', sheet_name='Base Resin', header=0, na_values='NaN')
        rbvalue = pd.read_excel('./model/PPvalue.xlsx', sheet_name='Rubber', header=0, na_values='NaN')

        count = 0
        # input PP
        while count != 4:
            inp = str(request.form['PP'+str(count+1)+'_name'])
            inp = inp.replace('p', 'P')
            inp = inp.replace('n', 'N')
            if inp == 'N':
                break
            elif inp[0] == 'P':
                if len(ppvalue[ppvalue['PP'] == inp]) == 0:
                    return render_template('algo.html')
                else:
                    inpdata.loc[0][3 * count +1] = ppvalue[ppvalue['PP'] == inp]['FM'].values[0]
                    inpdata.loc[0][3 * count +2] = ppvalue[ppvalue['PP'] == inp]['MI'].values[0]
                inpdata.loc[0][3 *count] = float(request.form['PP'+str(count+1)+'_rate'])
                count += 1
            else:
                return render_template('algo.html')
        count = 0
        # input Rubber
        while count != 4:
            inp = str(request.form['R'+str(count+1)+'_name'])
            inp = inp.replace('r', 'R')
            inp = inp.replace('n', 'N')
            if inp == 'N':
                break
            elif inp[0] == 'R':
                if len(rbvalue[rbvalue['RB'] == inp]) == 0:
                    return render_template('algo.html')
                else:
                    inpdata.loc[0][3 * count + 13] = rbvalue[rbvalue['RB'] == inp]['Density'].values[0]
                    inpdata.loc[0][3 * count + 14] = rbvalue[rbvalue['RB'] == inp]['Tensile Strength'].values[0]
                inpdata.loc[0][3 * count +12] = float(request.form['R'+str(count+1)+'_rate'])
                count += 1
            else:
                return render_template('algo.html')
                
        count = 0
        # input Filler
        while count != 3:
            inp = str(request.form['F'+str(count+1)+'_name'])
            inp = inp.replace('f', 'F')
            inp = inp.replace('n', 'N')
            if inp == 'N':
                break
            elif 'F001' <= inp <= 'F018':
                inpdata[inpdata.columns[inpdata.columns == inp]] = float(request.form['F'+str(count+1)+'_rate'])
                count += 1
            else:
                return render_template('algo.html')

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, './model/비중')
            outdata[0] = sess.run(hypothesis, feed_dict={X: inpdata})[0][0]

        # 굴곡탄성률 - XGB
        if os.path.exists('./model/굴곡탄성률.pkl'):
            load_model = joblib.load('./model/굴곡탄성률.pkl')
            outdata[1] = load_model.predict(inpdata)[0]

        # 굴곡강도 - SVR
        if os.path.exists('./model/굴곡강도.pkl'):
            load_model = joblib.load('./model/굴곡강도.pkl')
            outdata[2] = load_model.predict(inpdata)[0]

        # HDT - SVR
        if os.path.exists('./model/HDT.pkl'):
            load_model = joblib.load('./model/HDT.pkl')
            outdata[3] = load_model.predict(inpdata)[0]

        # IZOD - SVR
        if os.path.exists('./model/IZOD.pkl'):
            load_model = joblib.load('./model/IZOD.pkl')
            outdata[4] = load_model.predict(inpdata)[0]

        # MI - SVR
        if os.path.exists('./model/MI.pkl'):
            load_model = joblib.load('./model/MI.pkl')
            outdata[5] = load_model.predict(inpdata)[0]

        # 인장강도 - SVR
        if os.path.exists('./model/인장강도.pkl'):
            load_model = joblib.load('./model/인장강도.pkl')
            outdata[6] = load_model.predict(inpdata)[0]

        return render_template('algo.html', outdata = outdata)

@app.route('/pdpd')
def product():
    return render_template('pdpd.html')

@app.route('/edu')
def edu():
    return render_template('edu.html')

if __name__ == "__main__":
    app.run(debug=True)