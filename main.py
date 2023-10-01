
import numpy as np
import cv2
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
# print(e_w_g_c, nose_w_h_c, m_w_f_w_c)
score_ewg_d = round((1-(e_w_g_c-e_w_g_d)/(e_w_g_c-e_w_g_d+e_w_g_c-e_w_g_m))*100,1)
score_ewg_m = round((1-(e_w_g_c-e_w_g_m)/(e_w_g_c-e_w_g_d+e_w_g_c-e_w_g_m))*100,1)
score_nwh_d = round((1-(nose_w_h_c-nose_w_h_d)/(nose_w_h_c-nose_w_h_d+nose_w_h_c-nose_w_h_m))*100,1)
score_nwh_m = round((1-(nose_w_h_c-nose_w_h_m)/(nose_w_h_c-nose_w_h_d+nose_w_h_c-nose_w_h_m))*100,1)
score_mwfw_d = round((1-(m_w_f_w_c-m_w_f_w_d)/(m_w_f_w_c-m_w_f_w_d+m_w_f_w_c-m_w_f_w_m))*100,1)
score_mwfw_m = round((1-(m_w_f_w_c-m_w_f_w_m)/(m_w_f_w_c-m_w_f_w_d+m_w_f_w_c-m_w_f_w_m))*100,1)

image = cv2.imread(image_path_child)
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
org2 = (50, 90)
org3 = (50, 130)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
cv2.putText(image, f'eyes>> dad:{score_ewg_d}, mom:{score_ewg_m}', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
cv2.putText(image, f'nose>> dad:{score_nwh_d}, mom:{score_nwh_m}', org2, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
cv2.putText(image, f'mouth>> dad:{score_mwfw_d}, mom:{score_mwfw_m}', org3, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow('result Image', image) 
cv2.waitKey(0)
cv2.destroyAllWindows()