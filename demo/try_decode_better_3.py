import cv2
import numpy as np
from pyzbar.pyzbar import decode
import imutils

def detect_and_decode_qr(image_path):
    print("Resim yükleniyor...")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Resim yüklenemedi. Lütfen dosya yolunu kontrol edin.")
        return
    
    print("Resim gri tonlamalıya çevriliyor...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("Gaussian bulanıklaştırma uygulanıyor...")
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    print("Kenarlar tespit ediliyor...")
    edged = cv2.Canny(blurred, 50, 150, 255)
    
    print("Konturlar bulunuyor...")
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    print(f"{len(contours)} adet kontur bulundu.")
    
    qr_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            qr_contour = approx
            print("QR kodu olabilecek bir kontur bulundu.")
            break
    
    if qr_contour is not None:
        print("Perspektif düzeltmesi yapılıyor...")
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
        
        print("İyileştirilmiş görüntü işleniyor...")
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        print("QR kodu çözülüyor...")
        qr_codes = decode(warped_thresh)
        if qr_codes:
            for qr_code in qr_codes:
                qr_data = qr_code.data.decode('utf-8')
                print("QR Kodu İçeriği:", qr_data)
        else:
            print("QR kodu çözümlemesi başarısız oldu. Kod algılanamadı.")
        
        # İşlenmiş resmi göster
        cv2.imshow("Düzleştirilmiş QR Kodu", warped_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("QR kodu bulunamadı.")

# Resmin yolu
image_path = 'try_2.png'  # Resminizin yolunu buraya yazın

# QR kodunu tespit et ve çöz
detect_and_decode_qr(image_path)
