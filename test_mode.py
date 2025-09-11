

import os

model_paths = [
    "/group/amdneuralopt/huggingface/pretrained_models/microsoft/Phi-4-mini-instruct",
    "/group/amdneuralopt/huggingface/pretrained_models/microsoft/Phi-3-mini-4k-instruct",
    "/group/amdneuralopt/huggingface/pretrained_models/google/gemma-2-9b",
    "/group/amdneuralopt/huggingface/pretrained_models/google/gemma-3-270m",
    "/group/amdneuralopt/huggingface/pretrained_models/mistralai/Mistral-7B-v0.1",
    "/group/amdneuralopt/huggingface/pretrained_models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "/group/amdneuralopt/huggingface/pretrained_models/openai/gpt-oss-20b",
    "/group/amdneuralopt/huggingface/pretrained_models/allenai/OLMo-7B",
    "/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen1.5-0.5B",
    "/group/amdneuralopt/huggingface/hub/meta-llama/Llama-2-7b",
    "/group/amdneuralopt/huggingface/pretrained_models/meta-llama/Llama-3.2-3B-Instruct",
    "/group/amdneuralopt/huggingface/pretrained_models/grok_layer_num_1",          
    "/group/amdneuralopt/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B",
    "/group/amdneuralopt/huggingface/pretrained_models/Qwen/Qwen2.5-0.5B-Instruct",
    "/group/amdneuralopt/huggingface/pretrained_models/THUDM/chatglm3-6b"  
]

quant_schemes_p0 = [
    "w_int4_per_group_sym",
    "w_uint4_per_group_asym",
    "w_int4_per_group_asym",
    "w_int4_per_channel_sym",
    "w_uint4_per_channel_asym",
    "w_int4_per_channel_asym",
    "w_uint4_per_channel_sym",
]
gpu_num = 1

cmd_list = []
for quant_algo in ["awq", "gptq"]:
    for quant_scheme in quant_schemes_p0:
        for model_path in model_paths:
            os.makedirs("output", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            uuid = f"{model_path.split("/")[-1]}_{quant_scheme}"
            output_dir = f"output/{uuid}"
            log_file = f"logs/{uuid}.log"
            
            dataset = ""
            if "awq" == quant_algo:
                seq_len = 512
                dataset = "pileval_for_awq_benchmark"
            elif "gptq" == quant_algo:
                seq_len = 2048
                dataset = "wikitext_for_gptq_benchmark"
            else:
                print(f"quant_algo {quant_algo} not supported")
                exit(1)


            trust_remote_code_flag = "no_trust_remote_code"
            if "phi" in model_path.lower():
                trust_remote_code_flag = "no_trust_remote_code"
            else:
                trust_remote_code_flag = "trust_remote_code"

            data_type="bfloat16"

            cmd = f"log_file=$(pwd)/{log_file};cd /scratch/meng/tmp/Quark/quark/experimental/cli/;python main.py torch-llm-ptq --model_dir {model_path} --seq_len {seq_len} --output_dir {output_dir} --quant_scheme {quant_scheme} --num_calib_data 128 --quant_algo gptq --dataset {dataset} --model_export hf_format --data_type {data_type} --{trust_remote_code_flag} --exclude_layers  >> $log_file 2>&1;cd -"

            cmd_list.append(cmd)


    os.makedirs(quant_algo, exist_ok=True)

    gpu_cmd = [[] for k in range(gpu_num)]
    for k in range(len(cmd_list)):
        sgpu = f"export CUDA_VISIBLE_DEVICES={k%gpu_num}"
        gpu_cmd[k%gpu_num].append(f"{sgpu};{cmd_list[k]}\n")


    all_cmd = ["mkdir logs;"]
    for g in range(len(gpu_cmd)):
        file_name = f"gpu{g}_cmd.sh"
        all_cmd.append(f"sh ./{file_name};")
        with open(os.path.join(quant_algo, file_name),"w") as f:
            f.writelines(gpu_cmd[g])

    with open(os.path.join(quant_algo, "all_cmd.sh"),"w") as f:
        f.writelines(all_cmd)



