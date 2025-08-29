#!/usr/bin/env python3
"""
測試 OpenRouter 調用 Gemini 2.5 Flash Image Preview 進行圖片生成
"""
import requests
import json
import base64
import os
from datetime import datetime

# 你提供的 API keys
API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
]

def test_gemini_image_generation(api_key, prompt, use_free_version=True):
    """
    測試 Gemini 2.5 Flash Image Preview 圖片生成
    
    Args:
        api_key: OpenRouter API key
        prompt: 圖片生成提示詞
        use_free_version: 是否使用免費版本
    """
    
    # 選擇模型版本
    model = "google/gemini-2.5-flash-image-preview:free" if use_free_version else "google/gemini-2.5-flash-image-preview"
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/123hi123/tran3",
        "X-Title": "Gemini Image Test"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        print(f"🔄 測試 API Key: {api_key[:20]}...")
        print(f"📝 提示詞: {prompt}")
        print(f"🤖 模型: {model}")
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        print(f"📊 狀態碼: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # 打印完整回應以了解結構
            print("📋 完整回應:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 檢查是否有圖片相關內容
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                message = choice.get('message', {})
                content = message.get('content', '')
                
                print(f"💬 回應內容: {content}")
                
                # 檢查是否有圖片 URL 或 base64 資料
                if 'image_url' in message:
                    image_url = message['image_url']
                    print(f"🖼️  圖片 URL: {image_url}")
                    download_image(image_url, f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                
                # 檢查內容中是否包含圖片資訊
                if 'base64' in content.lower() or 'data:image' in content.lower():
                    print("🎨 發現可能的 base64 圖片資料")
                    save_base64_image(content)
                
                return True, result
            else:
                print("❌ 回應中沒有找到 choices")
                return False, result
                
        else:
            print(f"❌ 請求失敗: {response.status_code}")
            print(f"📄 錯誤回應: {response.text}")
            return False, response.text
            
    except Exception as e:
        print(f"💥 發生錯誤: {str(e)}")
        return False, str(e)

def download_image(url, filename):
    """下載圖片"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ 圖片已保存: {filename}")
        else:
            print(f"❌ 下載失敗: {response.status_code}")
    except Exception as e:
        print(f"💥 下載錯誤: {str(e)}")

def save_base64_image(content):
    """保存 base64 編碼的圖片"""
    try:
        # 尋找 base64 資料
        if 'data:image' in content:
            # 格式: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
            base64_data = content.split('base64,')[1] if 'base64,' in content else content
        elif 'base64' in content.lower():
            # 可能只是純 base64 字串
            lines = content.split('\n')
            base64_data = ''.join([line.strip() for line in lines if not line.strip().startswith('```')])
        else:
            return
            
        # 解碼並保存
        image_data = base64.b64decode(base64_data)
        filename = f"generated_image_b64_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        with open(filename, 'wb') as f:
            f.write(image_data)
        print(f"✅ Base64 圖片已保存: {filename}")
        
    except Exception as e:
        print(f"💥 Base64 解碼錯誤: {str(e)}")

def main():
    """主測試函數"""
    print("🚀 開始測試 OpenRouter Gemini 2.5 Flash Image Preview")
    print("=" * 60)
    
    # 測試提示詞
    test_prompts = [
        "Generate a beautiful sunset landscape with mountains and a lake, digital art style",
        "創建一幅賽博朋克風格的未來城市夜景圖片",
        "Generate an image of a cute cat sitting in a garden with flowers"
    ]
    
    # 選擇一個有效的 API key 來測試
    for i, api_key in enumerate(API_KEYS[:3]):  # 只測試前3個key
        print(f"\n🔑 測試 API Key #{i+1}")
        print("-" * 40)
        
        # 先測試免費版本
        for j, prompt in enumerate(test_prompts[:1]):  # 只測試第一個提示詞
            print(f"\n📝 測試提示詞 #{j+1}")
            success, result = test_gemini_image_generation(api_key, prompt, use_free_version=True)
            
            if success:
                print("✅ 測試成功！")
                break
            else:
                print("❌ 測試失敗，嘗試下一個...")
        
        if success:
            break
    
    print("\n🏁 測試完成")

if __name__ == "__main__":
    main()
