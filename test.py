import numpy as np
import cv2
from keras.models import load_model
# import scipy.io as sio
from dehaze_patchMap_dehaze import dehaze_patchMap
from PIL import Image
import matplotlib.image


def start_testing(base_path_hazyImg, base_path_result, imgname, save_dir, modelDir):
    print("Process image: ", imgname)

    image = cv2.imread(base_path_hazyImg + imgname + ".jpg")
    if image.shape[0] != 480 or image.shape[1] != 640:
        print('resize image tp 640*480')
        image = cv2.resize(image, (640, 480))
    hazy_input = np.reshape(image, (1, 480, 640, 3))
    model = load_model(modelDir)
    patchMap = model.predict(hazy_input, verbose=1)
    patchMap = np.reshape(patchMap, (480, 640))

    recover_result, tx = dehaze_patchMap(image, 0.95, patchMap)

    savename_result = save_dir + 'py_recover_' + str(imgname.split('.')[0]) + '.jpg'
    normalized_arr = (recover_result - np.min(recover_result)) / (np.max(recover_result) - np.min(recover_result))

    result_arr = normalized_arr.copy()
    matplotlib.image.imsave(savename_result, result_arr)

def start_testing_final_images(base_path_hazyImg, base_path_result, imgname, save_dir, modelDir):
    print("Process image: ", imgname)

    image = cv2.imread(base_path_hazyImg + imgname + ".jpg")
    if image.shape[0] != 480 or image.shape[1] != 640:
        print('resize image tp 640*480')
        image = cv2.resize(image, (640, 480))
    hazy_input = np.reshape(image, (1, 480, 640, 3))
    model = load_model(modelDir)
    patchMap = model.predict(hazy_input, verbose=1)
    patchMap = np.reshape(patchMap, (480, 640))

    recover_result, tx = dehaze_patchMap(image, 0.95, patchMap)

    savename_result = save_dir + 'py_recover_' + str(imgname.split('.')[0]) + '.jpg'
    normalized_arr = (recover_result - np.min(recover_result)) / (np.max(recover_result) - np.min(recover_result))

    result_arr = normalized_arr.copy()
    matplotlib.image.imsave(savename_result, result_arr)

if __name__ == "__main__":
    base_path_hazyImg = 'image'
    base_path_result = 'patchMap/'
    imgname = 'waterfall.tif'
    save_dir = 'result/'
    modelDir = 'PMS-Net.h5'
    start_testing(base_path_hazyImg, base_path_result, imgname, save_dir, modelDir)
