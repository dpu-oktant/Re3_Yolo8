import cv2
import numpy as np
from pyzbar.pyzbar import decode
import imutils

def detect_and_decode_qr(image_path):
    print("Resim yükleniyor...")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Resim yüklenemedi.")
        return
    
    print("Resim gri tonlamalıya çevriliyor...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("Histogram eşitleme yapılıyor...")
    equalized = cv2.equalizeHist(gray)
    
    print("Adaptif eşikleme yapılıyor...")
    thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    print("Kenarlar tespit ediliyor...")
    edged = cv2.Canny(thresh, 50, 200, 255)
    
    print("Konturlar bulunuyor...")
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    qr_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            qr_contour = approx
            break
    
    if qr_contour is not None:
        print("QR konturu bulundu, perspektif düzeltme yapılıyor...")
        rect = np.array([qr_contour[0][0], qr_contour[1][0], qr_contour[2][0], qr_contour[3][0]], dtype="float32")
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        cv2.imshow("sonuc:",warped)
        print("QR kodu çözülüyor...")
        qr_codes = decode(warped)
        for qr_code in qr_codes:
            qr_data = qr_code.data.decode('utf-8')
            print("QR Kodu İçeriği:", qr_data)
        
        cv2.imshow("Düzleştirilmiş QR Kodu", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("QR kodu bulunamadı.")

image_path = 'try_3.png'  # Resminizin yolunu buraya yazın
detect_and_decode_qr(image_path)
