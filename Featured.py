import cv2
import numpy as np

# 讀取圖片 dog.jpg，-1 表示讀取原始格式
ig2 = cv2.imread("dog.jpg", -1)
# 顯示原始圖片
cv2.imshow("original", ig2)

# 取得圖片的高度（width）與寬度（length）
width = ig2.shape[0]
length = ig2.shape[1]

# 將原始彩色圖片轉為灰階圖
ig = cv2.cvtColor(ig2, cv2.COLOR_BGR2GRAY)

# 使用大津法進行二值化處理
# ret 是閾值，th 是二值化後的結果
ret, th = cv2.threshold(ig, 0, 255, cv2.THRESH_OTSU)

# 侵蝕操作，用於去除小型雜點
k = np.ones((3, 3))  # 創建一個 3x3 的核
th = cv2.erode(th, k)  # 將二值化圖像進行侵蝕操作

# 補洞操作，用於填補二值化圖像中的內部孔洞
th2 = 255 - th  # 將圖像顏色反轉
# 計算連通組件，hole 是標記矩陣
_, hole = cv2.connectedComponents(th2)
for i in range(width):
    for j in range(length):
        # 將標記大於等於 2 的部分填充為白色（255）
        if hole[i, j] >= 2:
            th[i, j] = 255
# 顯示經過補洞操作後的二值化圖像
cv2.imshow("Binarization", th)

# 將原始圖片背景用白色填充（將對應於二值化圖像為黑色的部分填充白色）
for i in range(width):
    for j in range(length):
        if th[i, j] == 0:  # 如果二值化圖像的像素為黑色
            ig2[i, j] = 255  # 將原圖像素設為白色
# 顯示處理後的圖片
cv2.imshow("Back", ig2)

# 讀取另一張圖片 background.jpg
img = cv2.imread("background.jpg", -1)
# 將處理後的 ig2 圖片縮放至 80x50
ig2 = cv2.resize(ig2, (80, 50), interpolation=cv2.INTER_CUBIC)
# 更新縮放後的圖片尺寸
width = ig2.shape[0]
length = ig2.shape[1]

# 設定疊加圖片的左上角位置
X = 30  # 水平位置
Y = 68  # 垂直位置

# 將處理後的圖片疊加到另一張圖片上
for i in range(width):
    for j in range(length):
        # 如果 ig2 的當前像素不是白色，則將其疊加到 img
        if ig2[i, j, 0] != 255 or ig2[i, j, 1] != 255 or ig2[i, j, 2] != 255:
            img[Y + i, X + j] = ig2[i, j]

# 放大圖片
scale_factor = 1.5  # 放大比例
new_width = int(img.shape[1] * scale_factor)  # 計算放大後的寬度
new_height = int(img.shape[0] * scale_factor)  # 計算放大後的高度
img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # 進行放大

# 顯示放大後的圖片
cv2.imshow("image3", img)
# 將最終結果保存為 success.png
cv2.imwrite('success.png', img)
cv2.waitKey()