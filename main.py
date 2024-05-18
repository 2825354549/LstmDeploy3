import time

import numpy as np
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forcasting import pred
import os



app = Flask(__name__)
app.secret_key = '123456'

@app.route('/')
def home():
    # return render_template('vuess.html')
    # return render_template('home.html')
    return render_template('base.html')

# 定义一个上传文件的路由
@app.route('/upload_file', methods=['POST','GET'])
def upload_file():
    uploaded_file = request.files['file']
    # print(uploaded_file)
    if uploaded_file:
        # 保存上传的文件到服务器的某个目录
        upload_folder = 'static/csvFile'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, uploaded_file.filename)
        uploaded_file.save(file_path)

        # 将 file_path 存储在 session 中
        session['file_path'] = file_path

        print('交互成功！1111111')
        return jsonify({'message': '文件上传成功，正在进行分析！', 'file_path': file_path})
        # return render_template('pred.html')
        # return redirect(url_for('prediction'))

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    # 从 session 中获取 file_path
    start_time = time.time()
    pred_file_path = session.get('file_path')
    print(pred_file_path)
    pred_data = pred(pred_file_path)  # 假设 pred_data 是一个包含日期和OT数据的 DataFrame
    print(pred_data.head())

    # 转换为 JSON 格式并返回到前端
    # pred_json = pred_data.to_json(orient='split')
    data = {
        'date': pred_data['date'].tolist(),  # 假设 date 是日期数据的列
        'OT': pred_data['OT'].tolist()  # 假设 OT 是 OT 数据的列
    }
    print('cost time:', time.time() - start_time)
    return jsonify(data)
@app.route('/analysis',methods=['POST','GET'])
def analysis():
    with open('./static/results/result.txt', 'r') as file:
        contents = file.read()
    if not contents.strip():  # Check if the file is empty or contains only whitespace
        # If the file is empty, calculate mses and rmse from NumPy arrays
        preds = np.load('./static/results/pred.npy')
        trues = np.load('./static/results/true.npy')
        mses = []
        rmses = []
        for i in range(0, preds.shape[0], 24):
            pred = preds[i, :, :]
            true = trues[i, :, :]
            mse = mean_squared_error(true, pred)
            rmse = np.sqrt(mse)

            mses.append(float(mse))  # Convert NumPy float32 to native Python float
            rmses.append(float(rmse))  # Convert NumPy float32 to native Python float
        mse = str(np.mean(mses))
        rmse = str(np.mean(rmses))
    else:
        # If the file is not empty, read mses and rmse from the file
        lines = contents.split('\n')
        # print(lines[2])
        # print('8')
        mse = lines[1].split(":")[1]
        # print(line)
        mse = str(mse)
        rmse = lines[3].split(":")[1]
        # print(line)
        rmse = str(rmse)
    data = [
        {'Model': 'LSTM', 'RMSE': rmse, 'MSE': mse}
    ]
    return jsonify(data)


if __name__ == '__main__':
    app.run(port=9000,debug=True)
    # file = './static/csvFile/ETTm1.csv'
    # df = pred2(file)
