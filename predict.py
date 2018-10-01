from models import *
from utils import *

model = crnn(training=False)

def predict(img_dir, weights_dir):
    X = []
    model.load_weights(weights_dir)
    img = preprocess(img_dir)
    X.append(img)
    X = np.array(X)
    pred = model.predict(X, batch_size=1)

    print(ctc_decoder(pred))

predict('0916_Data Samples 2/0000_samples.png', 'checkpoints/weights-01-182.39')
