# User: 廖宇
# Date Development：2023/10/16 14:24
import time
import numpy as np
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from sklearn.metrics import mean_absolute_error, mean_squared_error
from forcasting import pred
import os
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from functools import partial
import socket
import time
import threading
from threading import Thread, Event
import eventlet
import json
import struct
from sqlalchemy import Column, Integer
from datetime import datetime
from datetime import datetime
from sqlalchemy import Column, Integer, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


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
    time1 = time.time()
    pred_data = pred(pred_file_path)
    # 假设 pred_data 是一个包含日期和OT数据的 DataFrame
    # print('cost time1',time.time()-time1)
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
        {'Model': 'vInformer', 'RMSE': rmse, 'MSE': mse}
    ]
    return jsonify(data)


app.config.from_object(__name__)
socketio = SocketIO(async_mode='eventlet')  # 创建socketio实例，eventlet异步模式
socketio.init_app(app, cors_allowed_origins='*')  # websocket通信跨域问题，*表示允许所有域名和端口的连接
CORS(app, resources={r'/*': {'origins': '*'}}, supports_credentials=True)  # vue与flask前后端通信的跨域，*表示允许来自任何来源的跨域请求



# 配置信息以连接数据库
HOSTNAME = "127.0.0.1"
PORT = 3306
USERNAME = "root"
PASSWORD = "password"
DATABASE = "mysql"
app.config['SQLALCHEMY_DATABASE_URI'] \
    = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"  # 配置数据库连接
db = SQLAlchemy(app)  # 创建SQLAlchemy数据库实例，并与Flask应用程序关联
# 定义数据库模型，并关联到test表格
class MysqlData(Base):
    __tablename__ = 'mysql'
    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    date = Column(DateTime, default=datetime.utcnow, nullable=False, comment='日期')
    measured_depth = Column(Float, nullable=False, comment='测量深度，表示钻头在井中的垂直距离')
    weight_on_bit = Column(Float, nullable=False, comment='钻压，表示钻头对井底的压力')
    average_standpipe_pressure = Column(Float, nullable=False, comment='平均立管压力，表示钻井液从泵出口到钻台的压力')
    average_surface_torque = Column(Float, nullable=False, comment='平均地面扭矩，表示钻杆在地面的转动力矩')
    rop = Column(Float, nullable=False, comment='钻进速度，表示钻头在井中的下降速度')
    average_rotary_speed = Column(Float, nullable=False, comment='平均转速，表示钻杆在井中的转动速度')
    mud_flow_in = Column(Float, nullable=False, comment='钻井液流入量，表示钻井液从泵入口到钻台的流量')
    diameter = Column(Float, nullable=False, comment='直径，表示钻头或钻杆的直径')
    average_hookload = Column(Float, nullable=False, comment='平均吊重，表示钻杆在井中的重量')
    hole_depth_tvd = Column(Float, nullable=False, comment='井深（真垂深），表示钻头在井中的垂直深度')
# 初始化数据库连接
# engine = create_engine('mysql://username:password@localhost:3306/database_name', echo=True)
engine = create_engine(f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4", echo=True)
# 创建数据表
Base.metadata.create_all(engine)

# 创建DBSession类型:
DBSession = sessionmaker(bind=engine)


# 存储数据到mysql
def data_to_database(data):
    try:
        # 创建一个数据库会话
        session = DBSession()
        # 创建一个新的MysqlData对象
        new_data = MysqlData(date=data[0], measured_depth=data[1], weight_on_bit=data[2],
                             average_standpipe_pressure=data[3],
                             average_surface_torque=data[4], rop=data[5], average_rotary_speed=data[6],
                             mud_flow_in=data[7],
                             diameter=data[8], average_hookload=data[9], hole_depth_tvd=data[10])
        # 添加到数据库会话
        session.add(new_data)
        # 提交数据
        session.commit()
        # 关闭数据库会话
        session.close()
    except Exception as e:
        print(f"Failed to store data in database: {str(e)}")

if __name__ == '__main__':
    app.run(port=9000,debug=True)




