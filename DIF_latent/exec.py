import subprocess
#["caiso"],[10],[720]
#["ETTh1"],[7],[168]
#["production"],[18],[720]
#["ETTm1"],[7],[168],[192]
#{"solar"],[137],  

num = 0
for seq_len in [512]:
    for pred_len in [96,192,336,720]:
        for dataset_name,num_vars in zip(["ETTh1"],[7]):
            for embed_dim,num_heads in zip([256,256,256,128,128,128],[16,8,4,16,8,4]):
                for patch_len,stride in zip([8,16,32],[4,8,16]):
                    for mlp_ratio in [1.,2.,3.,4.]:
                        num += 1
                        cmd = f'python -u main_ddpm.py --seq_len {seq_len} --pred_len {pred_len} --label_len 24 --dataset_name {dataset_name} --features M --num_vars {num_vars} --pretrain_epochs 0 --train_epochs 100 --batch_size 64 --test_batch_size 64 --devices 0 --embed_dim {embed_dim} --num_heads {num_heads} --patch_len {patch_len} --stride {stride} --mlp_ratio {mlp_ratio}'
                        try:
                            print(num)
                            f = open("result.txt", 'a')
                            f.write(cmd)
                            f.write('\n')
                            f.close()
                            subprocess.run(cmd, check=True, shell=True)
                            print('process done')
                        except Exception as e:
                            print(f"{e}")

