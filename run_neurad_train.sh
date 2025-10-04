# CUDA_VISIBLE_DEVICES=1 python nerfstudio/scripts/train.py unisim --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "05fa5048-f355-3274-b565-c0ddc547b315" --data /workspace/datasets/self-driving/argoverse2

CUDA_VISIBLE_DEVICES=0 python nerfstudio/scripts/train.py unisim --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "0b86f508-5df9-4a46-bc59-5b9536dbde9f" --data /workspace/datasets/self-driving/argoverse2

CUDA_VISIBLE_DEVICES=0 python nerfstudio/scripts/train.py unisim --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "185d3943-dd15-397a-8b2e-69cd86628fb7" --data /workspace/datasets/self-driving/argoverse2

CUDA_VISIBLE_DEVICES=0 python nerfstudio/scripts/train.py unisim --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "25e5c600-36fe-3245-9cc0-40ef91620c22" --data /workspace/datasets/self-driving/argoverse2

CUDA_VISIBLE_DEVICES=0 python nerfstudio/scripts/train.py unisim --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "27be7d34-ecb4-377b-8477-ccfd7cf4d0bc" --data /workspace/datasets/self-driving/argoverse2
