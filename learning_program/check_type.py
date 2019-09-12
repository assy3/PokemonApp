def main(img_path):    
    from keras import models
    import numpy as np
    from keras.preprocessing import image
    from keras.models import model_from_json

    types = ["ノーマル", "ほのお", "みず", "くさ", "でんき", "こおり", "かくとう", "どく", "じめん", "ひこう", "エスパー", "むし", "いわ", "ゴースト", "ドラゴン"]

    model = model_from_json(open('./learned_data/data.json').read())
    model.load_weights('./learned_data/data.hdf5')

    img_path = img_path 
    img = image.load_img(img_path, target_size=(240,240,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    features = model.predict(x)

    for i in range(0,14):
        if features[0,i] == 1:
            break
    
    return types[i]


if __name__ == '__main__':
    result = main('test.jpg')
    print(result)