from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
import argparse
import uuid
from img_process import get_test
import json
from datetime import timedelta

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        imgName=str(uuid.uuid1())
        cv2.imwrite(os.path.join(basepath, 'static/images', imgName+'.jpg'), img)
        result=get_test(imgName)
        dict={"生理性推荐侧卧枕高":result[0],"生理性推荐仰卧枕高":result[1]}
        return json.dumps(dict,ensure_ascii=False)
        return render_template('upload_ok.html', userinput=user_input, val1=time.time())

    return render_template('upload.html')

def get_parse(imgName):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='static/images/'+imgName+'.jpg', help='my photo')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8987, debug=True)