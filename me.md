id,sign_language,nose_x,nose_y,nose_z,left_shoulder_x,left_shoulder_y,left_shoulder_z,right_shoulder_x,right_shoulder_y,right_shoulder_z,left_hand_0_x,left_hand_0_y,left_hand_0_z,left_hand_1_x,left_hand_1_y,left_hand_1_z,left_hand_2_x,left_hand_2_y,left_hand_2_z,left_hand_3_x,left_hand_3_y,left_hand_3_z,left_hand_4_x,left_hand_4_y,left_hand_4_z,left_hand_5_x,left_hand_5_y,left_hand_5_z,left_hand_6_x,left_hand_6_y,left_hand_6_z,left_hand_7_x,left_hand_7_y,left_hand_7_z,left_hand_8_x,left_hand_8_y,left_hand_8_z,left_hand_9_x,left_hand_9_y,left_hand_9_z,left_hand_10_x,left_hand_10_y,left_hand_10_z,left_hand_11_x,left_hand_11_y,left_hand_11_z,left_hand_12_x,left_hand_12_y,left_hand_12_z,left_hand_13_x,left_hand_13_y,left_hand_13_z,left_hand_14_x,left_hand_14_y,left_hand_14_z,left_hand_15_x,left_hand_15_y,left_hand_15_z,left_hand_16_x,left_hand_16_y,left_hand_16_z,left_hand_17_x,left_hand_17_y,left_hand_17_z,left_hand_18_x,left_hand_18_y,left_hand_18_z,left_hand_19_x,left_hand_19_y,left_hand_19_z,left_hand_20_x,left_hand_20_y,left_hand_20_z,right_hand_0_x,right_hand_0_y,right_hand_0_z,right_hand_1_x,right_hand_1_y,right_hand_1_z,right_hand_2_x,right_hand_2_y,right_hand_2_z,right_hand_3_x,right_hand_3_y,right_hand_3_z,right_hand_4_x,right_hand_4_y,right_hand_4_z,right_hand_5_x,right_hand_5_y,right_hand_5_z,right_hand_6_x,right_hand_6_y,right_hand_6_z,right_hand_7_x,right_hand_7_y,right_hand_7_z,right_hand_8_x,right_hand_8_y,right_hand_8_z,right_hand_9_x,right_hand_9_y,right_hand_9_z,right_hand_10_x,right_hand_10_y,right_hand_10_z,right_hand_11_x,right_hand_11_y,right_hand_11_z,right_hand_12_x,right_hand_12_y,right_hand_12_z,right_hand_13_x,right_hand_13_y,right_hand_13_z,right_hand_14_x,right_hand_14_y,right_hand_14_z,right_hand_15_x,right_hand_15_y,right_hand_15_z,right_hand_16_x,right_hand_16_y,right_hand_16_z,right_hand_17_x,right_hand_17_y,right_hand_17_z,right_hand_18_x,right_hand_18_y,right_hand_18_z,right_hand_19_x,right_hand_19_y,right_hand_19_z,right_hand_20_x,right_hand_20_y,right_hand_20_z

same id is same video 並且已經按照 順序排好
名稱上 等於 sord.csv 的 sign_language 的 去看它下面的value 如果是 s 則要評估一個video 他的 左手 右手 缺失值比例 缺失達到80趴以上的就刪去全部資料 那個手的全部資料
然後 處理 鼻子缺失 與肩膀 用前後的值填充
第一個 就一直往後找 直到找到 不是0的值 填入
最後一個 就一直往前找 直到找到 不是0的值 填入
中間的 就取前後找到的第一個平均值 填入
至於手部不插值
    語意重要性：
    - 手是手語的核心，錯誤的插值可能改變語意。

    運動複雜性：
    - 手部運動模式比頭部、肩膀複雜得多。

    缺失原因多樣：
    - 可能是有意的手勢變化，不一定是技術問題。
都處理完之後填充 缺失值 用 0 填充
設計兩個啟動參數 有low 那就是一次載入一組（同id) 處理完畢在下一組
沒有啟動參數 那就是
座標 get by the mediapipe
然後我在想使用stgcn 來訓練 你感覺如何 不錯的話 要怎麼清理我們的資料集


要有一個 gen.py 用來生成 虛假數據 提供測試
然後你的 clean.py 我都會用 --low 來啟動 確保 low 啟動參數相關設置已經寫好
clean 後的東西會輸出到 output/ 底下



