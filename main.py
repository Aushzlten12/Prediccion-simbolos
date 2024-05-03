import base64
import io

import numpy as np
from flask import Flask, jsonify, redirect, request
from PIL import Image
from skimage.transform import resize
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cargar el modelo durante la inicialización de la aplicación Flask
model = load_model("modelo.h5")

main_html = """
<html>
<head></head>
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;

  function InitThis() {
      ctx = document.getElementById('myCanvas').getContext("2d");

      symbols = ["♥", "♦", "♣", "♠"];
      mensaje_symbols = symbols.join(",")      
      document.getElementById('mensaje').innerHTML  = 'Dibuja uno de estos símbolos ' + mensaje_symbols;

      $('#myCanvas').mousedown(function (e) {
          mousePressed = true;
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
      });

      $('#myCanvas').mousemove(function (e) {
          if (mousePressed) {
              Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
          }
      });

      $('#myCanvas').mouseup(function (e) {
          mousePressed = false;
      });
  	    $('#myCanvas').mouseleave(function (e) {
          mousePressed = false;
      });
  }

  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 11;
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.closePath();
          ctx.stroke();
      }
      lastX = x; lastY = y;
  }

  function clearArea() {
      // Use the identity matrix while clearing the canvas
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
  function prepareImg() {
     var canvas = document.getElementById('myCanvas');
     document.getElementById('myImage').value = canvas.toDataURL();
  }

</script>
<body onload="InitThis();">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript" ></script>
    <div align="left">
      <img src="https://upload.wikimedia.org/wikipedia/commons/f/f7/Uni-logo_transparente_granate.png" width="300"/>
    </div>
    <div align="center">
        <h1 id="mensaje">Dibujando...</h1>
        <canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
        <br/>
        <br/>
        <button onclick="javascript:clearArea();return false;">Borrar</button>
    </div>
    <div align="center">
      <form method="post" action="predict" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
      <input id="myImage" name="myImage" type="hidden" value="">
      <input id="bt_upload" type="submit" value="Predecir">
      </form>
    </div>
</body>
</html>
"""


@app.route("/")
def main():
    return main_html


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_data = request.form.get("myImage").replace("data:image/png;base64,", "")
        img = Image.open(io.BytesIO(base64.b64decode(img_data)))
        img = img.split()[3]  # Tomar solo la cuarta capa de la imagen RGBA

        size = (28, 28)
        img = np.array(img) / 255.0
        img = resize(img, size)
        img = np.expand_dims(img, axis=0)

        # Hacer la predicción utilizando el modelo previamente cargado
        prediction = model.predict(img)
        predicted_symbol = np.argmax(prediction)

        int_to_symbol = {0: "♥", 1: "♦", 2: "♣", 3: "♠"}
        name_symbol = int_to_symbol[predicted_symbol]

        percentage_similarity = {
            symbol: f"{prob * 100:.2f}%"
            for symbol, prob in zip(int_to_symbol.values(), prediction[0])
        }
        # Formatear los resultados como JSON
        results = {
            "El simbolo se parece a": name_symbol,
            "Porcentaje de similitud": percentage_similarity,
        }

        return jsonify(results)
    except Exception as e:
        print("Error occurred:", e)
        return redirect("/", code=302)


if __name__ == "__main__":
    app.run()
