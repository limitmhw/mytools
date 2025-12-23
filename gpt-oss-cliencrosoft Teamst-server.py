import requests
import json

SERVER_IP = "xcoh100-1"
PORT = 9090

url = f"http://{SERVER_IP}:{PORT}/v1/chat/completions"

payload = {
    "model": "/scratch/meng/gpt-oss-120b",  # 注意：需要确认模型名是否正确
    "messages": [
        {
            "role": "user",
            "content": "你是谁？你会不会写小说，能不能写一个仙侠题材的小说大纲"
        }
    ]
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    result = response.json()
    # 输出模型生成的内容
    message = result["choices"][0]["message"]["content"]
    print("模型回答:", message)
else:
    print("请求失败，状态码:", response.status_code)
    print(response.text)


# export VLLM_USE_DEEP_GEMM=0
# export VLLM_ENABLE_FP8=0

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# vllm serve /scratch/meng/gpt-oss-120b \
#   --tensor-parallel-size 4 \
#   --dtype bfloat16 \
#   --disable-custom-all-reduce \
#   --host 0.0.0.0 \
#   --port 9090




