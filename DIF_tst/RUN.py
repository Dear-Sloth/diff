import subprocess
#["caiso"],[10],[720]
#["ETTh1"],[7],[168]
#["production"],[18],[720]
#["ETTm1"],[7],[168],[192]
#{"solar"],[137],  
#import torch
#print(torch.version)
num = 0
for dataset_name,num_vars in zip(["ETTh1","ETTm1","ETTh2","ETTm2"],[7,7,7,7]):
    for seq_len in [512]:
        for pred_len in [96,192,336,720]:
            num += 1
            cmd = f'python -u main_ddpm.py --seq_len {seq_len} --pred_len {pred_len}  \
            --dataset_name {dataset_name} --features M --num_vars {num_vars} \
            --e_layers 3 \
            --n_heads 4 \
            --d_model 16 \
            --d_ff 128 \
            --dropout 0.3\
            --fc_dropout 0.3\
            --head_dropout 0\
            --patch_len 16\
            --stride 8\
            --pretrain_epochs 0 --train_epochs 100 --batch_size 64 --test_batch_size 64 --devices 2'
            try:
                print(num)
                subprocess.run(cmd, check=True, shell=True)
                print('process done')
            except Exception as e:
                print(f"{e}")

