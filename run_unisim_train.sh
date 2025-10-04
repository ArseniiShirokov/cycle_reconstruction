# CUDA_VISIBLE_DEVICES=1 python nerfstudio/scripts/train.py neurad --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "05fa5048-f355-3274-b565-c0ddc547b315" --data /workspace/datasets/self-driving/argoverse2

CUDA_VISIBLE_DEVICES=1 python nerfstudio/scripts/train.py neurad --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "0b86f508-5df9-4a46-bc59-5b9536dbde9f" --data /workspace/datasets/self-driving/argoverse2

CUDA_VISIBLE_DEVICES=1 python nerfstudio/scripts/train.py neurad --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "185d3943-dd15-397a-8b2e-69cd86628fb7" --data /workspace/datasets/self-driving/argoverse2

CUDA_VISIBLE_DEVICES=1 python nerfstudio/scripts/train.py neurad --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "25e5c600-36fe-3245-9cc0-40ef91620c22" --data /workspace/datasets/self-driving/argoverse2

CUDA_VISIBLE_DEVICES=0 python nerfstudio/scripts/train.py neurad --pipeline.datamanager.num_processes 0 --experiment_name="our_split_av2" argoverse2-data --sequence "5589de60-1727-3e3f-9423-33437fc5da4b" --data /workspace/datasets/self-driving/argoverse2
