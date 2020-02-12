from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
#programı her defasında farklı modlarla,farklı sonuçlar üretecek şekilde başlatmak için kullanılır.
#argparse kütüphanesi ile bir python programına gelen argümanları kontrol altına alabiliriz.

import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image", default='images/test_01.png')
#-i şeklinde yazarak sıralı olma kuralından kurtulmuş olduk.
#Eğer istersek --isim şeklinde uzun yazmak yerine, -i gibi kısa alternatif tanımlayabiliriz

#--------HELP---------------------------------------------------------------------
#Programımıza -h şeklinde argüman gönderirsek, bize argümanlar hakkında yardım mesajı gösterecektir.
#Yardım mesajlarını tanımlamak için help parametresini kullanabiliriz
#required kısmı bunun gerekli bir argüman olup olmadığını belirtiyor
#eğer True yaparsanız bu değer girilmeden program başlamıyacaktır.

args = vars(ap.parse_args()) # Gelen argümanları ayırıyoruz ve bir değişkene aktarıyoruz

ANSWER_KEY = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1,5: 0,6: 1,7: 2,8: 1,9: 0,10: 2,11: 0, 12: 3, 13: 2,14: 2,15: 0}

image = cv2.imread(args["image"]) #resmimizi cv2 ile taradık

#Canny de daha iyi sonuç almak için grileştirelim
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Kenar tespiti görüntüdeki kirliliğe karşı hassas olduğu için,
#ilk adım görüntüdeki kirliliği 5x5 Gaussian filter ile kaldırmaktır
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 70, 90)  #kenar algılama algoritması


#Konturlar, aynı renk veya yoğunluğa sahip olan tüm kesintisiz noktaları 
#birleştiren bir eğri. cv2.findContours() işleminde konturlar bulunur
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 #1.Birinci argüman kontur bulunacak kaynak görüntüdür.
 #2.İkinci argüman kontur alma modudur. 
 #3.Üçüncü argüman ise kontur yaklaşım metodur.
cnts = imutils.grab_contours(cnts) #kontürleri yakalar
docCnt = None

#en az kontür alınacak 1 tane alan bulunmalı
#kontürleri büyükten küçüğe sıralarız.
#sıralanmış kontürleri döngüye sokarız
# kontürde dört nokta var ise kağıdın ana hatlarını bulduk demektir.
#Bunları if in içine yazınca girinti hatası veriyo!
if len(cnts) > 0:
	
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	for c in cnts:
	
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			docCnt = approx
			break
        
#perspektif dönüşümü uygular.keser optiği dört kenarından.
#paper = four_point_transform(image, docCnt.reshape(4, 2))
#warped = four_point_transform(gray, docCnt.reshape(4, 2))
paper = image
warped = gray

thresh = cv2.threshold(warped, 0, 205,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = [] #yuvarlaklara karşılık gelen kontürlerin listesi oluşturulur.
#görüntünün hangi bölgelerinin yuvarlak olduğunu belirlemek için döngü
for c in cnts:
    #bu kontürlerin her biri için  en boy oranını hesaplamamıza yardımcı olacak
    #bounding box ı oluşturalım.
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h) #en boy oranını hesaplar

    #genişlik ve uzunluk en az kaç piksel olmalı
    #daire olacağı içinde oranları(ar) yaklaşık 1 e yakın olmalı
	if w >= 10 and h >= 10 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0] #burda hata verirse weight ve height i değiştir.
    #yukardan aşağıya sıralarız
correct = 0
#doğru sayısını bulmak için kullanıcağımız sayaç

for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
    #her sorunun 4 olası cevabı var
	#4lü 4lü grupladık bir satırı almak için ve her bir grubu döngüye aldık.
    
    #soldan sağa sıraladık.
	cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
	bubbled = None


	for (j, c) in enumerate(cnts):
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, 3)


		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)
        
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	color = (255, 0, 200) #doğru olan cevabın rengini mor olarak seçer.
	k = ANSWER_KEY[q]

	if k == bubbled[1]:
		color = (0, 255, 0) #eğer öğrenci de doğru yapmış ise baloncuk rengini yeşil seçer
		correct += 1

	cv2.drawContours(paper, [cnts[k]], -1, color, 3) #baloncuğu seçilen renge boyar.

score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 205), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)