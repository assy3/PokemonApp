import os
import json
# request フォームから送信した情報を扱うためのモジュール
# redirect  ページの移動
# url_for アドレス遷移
from flask import Flask, request, redirect, url_for
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
# 画像のダウンロード
from flask import send_from_directory

from keras import models
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from keras import backend as K

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def gosanke_func(img_path):
    K.clear_session()
    gosanke =("フシギダネ", "ヒトカゲ", "ゼニガメ",
          "チコリータ", "ヒノアラシ", "ワニノコ",
          "キモリ", "アチャモ", "ミズゴロウ",
          "ナエトル", "ヒコザル", "ポッチャマ",
          "ツタージャ", "ポカブ", "ミジュマル",
          "ハリマロン", "フォッコ", "ケロマツ",
          "モクロー", "ニャビー", "アシマリ",
          "サルノリ", "ヒバニー", "メッソン")

    model = model_from_json(open('./learned_data_gosanke/data_gosanke.json').read())
    model.load_weights('./learned_data_gosanke/data_gosanke.hdf5')

    img_path = img_path 
    img = image.load_img(img_path, target_size=(120,120,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    features = model.predict(x)
    i = np.argmax(features[0])

    output = gosanke[i] 

    return output
    
def type_func(img_path):    
    K.clear_session()
    types = ("ノーマル", "ほのお", "みず", "くさ", "でんき", "こおり", "かくとう", "どく", "じめん", "ひこう", "エスパー", "むし", "いわ", "ゴースト", "ドラゴン")

    model = model_from_json(open('./learned_data/data.json').read())
    model.load_weights('./learned_data/data.hdf5')

    img_path = img_path 
    img = image.load_img(img_path, target_size=(120,120,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    features = model.predict(x)
    i = np.argmax(features[0])

    output = types[i] 
    
    return output
    
def result_func(filepath):
    gosanke_result = gosanke_func(filepath)
    type_result = type_func(filepath)
    return gosanke_result, type_result


def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # ファイルを受け取る方法の指定
@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            # flash('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            # flash('ファイルがありません')
            return redirect(request.url)
        # ファイルのチェック
        if file and allwed_file(file.filename):
            # 危険な文字を削除（サニタイズ処理）
            global filename
            filename = secure_filename(file.filename)
            # ファイルの保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # アップロード後のページに転送
            print(file.filename)
            print(filename)
            
            # 関数呼び出し ポケモンの属性を文字で返す
            global target_path 
            target_path = "./uploads/" + filename

            global result_name
            global result_type
            result_name, result_type = result_func(target_path)
            
            return redirect(url_for('result'))

    return '''
    <!doctype html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>
                ポケモンタイプ診断
            </title>
            <link rel="stylesheet" href="./uploads/toppage.css">
        </head>

        <body> 
            <img src="./uploads/pokemon.jpg">
            <h1>
                ファイルをアップロードしてポケモンのタイプを判定しよう
            </h1>
            <ul>
                <li><b>好きな画像をアップロードしてみよう！</b></li>
                <li><b>その画像が何タイプなのかAIが教えてくれるよ</b></li>
                <li><b>タイプは全部で15種類！</b></li>
                <li><b>Let's Try!</b></li>
            </ul>
            <form method = post enctype = multipart/form-data>
            <p><input type=file class="q" name = file>
            <input type = submit class="qButton" value = 調べる>
            <div id="result"></div>
            </form>

            <script>
                const qs = (q) => document.querySelector(q)
                window.onload = () => {
                const q = qs('#q')
                const qButton = qs('#qButton')
                const result = qs('#result')
                // 判定ボタンを押した時 --- (*1)
                qButton.onclick = () => {
                    result.innerHTML = "..."
                    // APIサーバに送信するURLを構築 --- (*2)
                    const api = "/api?q=" + 
                    encodeURIComponent(q.value)
                    // APIにアクセス --- (*3)
                    fetch(api).then((res) => {
                    return res.json() // JSONで返す
                    }).then((data) => {
                    // 結果を画面に表示 --- (*4)
                    result.innerHTML =
                        data["label"] + 
                        "<span style='font-size:0.5em'>(" + 
                        data["per"] + ")</span>"
                    })
                }
                }
            </script>
        </body>
'''

@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# resultした場合 
@app.route('/result', methods=['GET'])
def result():
    result_format = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><title>ポケモンタイプ</title>
    <link rel="stylesheet" href="./uploads/resultpage.css">
    </head>
    <body>
        <div align="center" class="main">
            <img src="{0}">
            <p>
                <h1>結果画面</h1>
                <h2>この画像は<font color="red">{2}</font>タイプのポケモンぽいです</h2>

                <h3>この画像はポケモン御三家の中では<font color="red">{1}</font>に似てます</h3>
            </p>
<<<<<<< HEAD
            <b><a href="/" class="btn-gradient-3d-orange">もう一度調べる</a><br></b>

            <b><p>結果をSNSでシェアする</p></b>
            
            <!--twitter-->
            <a class="btn-social-square btn-social-square--twitter">
            <i class="fab fa-twitter"></i>
            </a>
            <!--facebook-->
            <a class="btn-social-square btn-social-square--facebook">
            <i class="fab fa-facebook"></i>
            </a>
            <!--はてぶ-->
            <a class="btn-social-square btn-social-square--hatebu">
            B!
            </a>
            <!--pocket-->
            <a class="btn-social-square btn-social-square--pocket">
            <i class="fab fa-get-pocket"></i>
            </a>
            <!--feedly-->
            <a class="btn-social-square btn-social-square--feedly">
            <i class="fas fa-rss"></i>
            </a>
=======
            <a href="/">もう一度調べる!</a><br>
>>>>>>> 80e37375e9b54ac318ccc7af126d818d31949d16
        </div>
    </body>
    </html>
    """.format(target_path, result_name, result_type)
    return result_format 

if __name__ == '__main__':
    app.run(port=6006)