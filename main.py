
import numpy as np
from utils import init_lmark_model, get_single_image_stats
# reference: https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# reference: https://github.com/davisking/dlib-models
image_path_mom = "samples/gigi-hadid.jpeg"
image_path_child = "samples/gigi-hadid-daughter.jpg"
image_path_dad = "samples/gigi-hadid-husband.jpg"

detector, predictor = init_lmark_model()

e_w_g_m, nose_w_h_m, m_w_f_w_m = get_single_image_stats(image_path_mom, detector, predictor)
e_w_g_d, nose_w_h_d, m_w_f_w_d = get_single_image_stats(image_path_dad,detector, predictor)
e_w_g_c, nose_w_h_c, m_w_f_w_c = get_single_image_stats(image_path_child,detector, predictor)
print(e_w_g_c, nose_w_h_c, m_w_f_w_c)
