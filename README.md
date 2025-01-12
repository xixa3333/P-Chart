# 圖像處理工具
此專案利用 OpenCV 完成多種圖像處理功能，包括灰階轉換、二值化、背景填充、疊加圖片與放大處理，最終生成處理後的圖片結果。
這是我們關於影像處理微學分的期末小專題。
我在此報告負責撰寫python。

## 功能
1. **灰階轉換**：將原始彩色圖片轉換為灰階圖像。
2. **二值化**：使用大津法進行二值化處理，並進行侵蝕操作和補洞處理。
3. **背景填充**：將原圖的背景填充為白色。
4. **圖片疊加**：將處理後的圖片疊加到另一張背景圖片上。
5. **圖片放大**：將疊加後的圖片放大至設定比例。
6. **圖片輸出**：將最終處理結果保存為 `success.png`。

## 文件結構
解壓縮 ZIP 檔案後，包含以下內容：
- `process.py`：主要程式碼。
- `dog.jpg`：用於處理的原始圖片。
- `background.jpg`：用於疊加的背景圖片。

## 必要條件
- Python 3.8 或以上版本。
- 已安裝 OpenCV 和 NumPy：
  ```bash
  pip install opencv-python-headless numpy
  ```

## 使用方式
1. **下載並解壓 ZIP 檔案**：
   解壓後確認內容完整。

2. **執行程式**：
   在終端機或命令提示字元中執行以下指令：
   ```bash
   python process.py
   ```

3. **檢視處理結果**：
   - 執行後會彈出多個視窗顯示圖片的不同處理階段：
     - 原始圖片。
     - 二值化後的圖片。
     - 背景填充後的圖片。
     - 最終放大處理後的圖片。
   - 處理完成後，最終結果將保存為 `success.png`。

## 注意事項
1. 請確保目錄下存在 `dog.jpg` 和 `background.jpg`，否則程式無法執行。
2. 程式中預設將處理後的圖片縮放為 `(80, 50)` 並疊加到背景圖片的位置 `(X=30, Y=68)`。若需更改，請修改程式碼中的相關變數。
3. 使用 OpenCV 的 GUI 功能會彈出圖片視窗，請按下任意鍵關閉。

## 範例輸出
最終圖片範例（`success.png`）將包含以下特徵：
- 原始圖片經處理後的結果。
- 疊加到背景圖片中的特定區域。
- 放大至設定比例的最終效果。

## 聯絡方式
若有問題或改進建議，請聯繫作者。
