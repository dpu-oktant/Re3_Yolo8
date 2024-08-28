import cv2
import numpy as np
from pyzbar.pyzbar import decode

def optimize_and_decode_qr(image_path):
    print("Resim yükleniyor...")
    image = cv2.imread(image_path)

    if image is None:
        print("Resim yüklenemedi. Lütfen dosya yolunu kontrol edin.")
        return

    print("Resim gri tonlamalıya çevriliyor...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Hafif keskinleştirme uygulanıyor...")
    kernel_sharpening = np.array([[0, -0.5, 0], 
                                  [-0.5, 3, -0.5],
                                  [0, -0.5, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel_sharpening)

    print("Kenarlar tespit ediliyor...")
    edged = cv2.Canny(sharpened, 30, 100)

    print("Konturlar bulunuyor...")
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"{len(contours)} adet kontur bulundu.")

    qr_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > 500:
            qr_contour = approx
            print("QR kodu olabilecek bir kontur bulundu.")
            break

    if qr_contour is not None:
        print("Perspektif düzeltmesi yapılıyor...")
        rect = np.array([qr_contour[0][0], qr_contour[1][0], qr_contour[2][0], qr_contour[3][0]], dtype="float32")

        width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
        height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))

        dim = int(max(width, height))

        dst = np.array([
            [0, 0],
            [dim - 1, 0],
            [dim - 1, dim - 1],
            [0, dim - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (dim, dim))

        # Perspektif düzeltmesinden sonra netleştirme ve bulanıklık azaltma işlemi
        print("Netleştirme ve bulanıklık azaltma uygulanıyor...")
        kernel_sharpening_post = np.array([[0, -1, 0], 
                                           [-1, 5, -1],
                                           [0, -1, 0]])
        final_warped = cv2.filter2D(warped, -1, kernel_sharpening_post)

        # Daha fazla keskinleştirme ve bulanıklık azaltma
        print("Daha fazla keskinleştirme uygulanıyor...")
        kernel_sharpening_more = np.array([[0, -1, 0], 
                                           [-1, 6, -1],
                                           [0, -1, 0]])
        final_warped = cv2.filter2D(final_warped, -1, kernel_sharpening_more)

        print("Gürültü azaltma uygulanıyor...")
        final_warped = cv2.fastNlMeansDenoisingColored(final_warped, None, 10, 10, 7, 21)

        print("Min filter azaltılıyor...")
        final_warped = cv2.erode(final_warped, np.ones((2, 2), np.uint8))

        print("Kontrast artırma uygulanıyor...")
        final_warped_gray = cv2.cvtColor(final_warped, cv2.COLOR_BGR2GRAY)
        final_warped_gray = cv2.equalizeHist(final_warped_gray)

        print("Adaptif eşikleme uygulanıyor...")
        final_warped_thresh = cv2.adaptiveThreshold(final_warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        print("QR kodu çözülüyor...")
        qr_codes = decode(final_warped_thresh)
        if qr_codes:
            for qr_code in qr_codes:
                qr_data = qr_code.data.decode('utf-8')
                print("QR Kodu İçeriği:", qr_data)
        else:
            print("QR kodu çözümlemesi başarısız oldu. Kod algılanamadı.")

        cv2.imshow("Düzleştirilmiş QR Kodu", final_warped_thresh)
        cv2.imwrite("/mnt/data/optimized_qr.png", final_warped_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("QR kodu bulunamadı.")

# Daha önce elde edilen görselin yolu
image_path = 'try_2.png'

# QR kodunu tespit et ve çöz
optimize_and_decode_qr(image_path)
