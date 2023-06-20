
CUDA_VISIBLE_DEVICES=0 nohup python run_zeroshot_cls.py --model "ViT-B/32" --dataset "imagenet" --image_emb_type "mask2former_context" --batch_size 32 --num_samples -1 > imagenet_logs.txt &
CUDA_VISIBLE_DEVICES=1 nohup python run_zeroshot_cls.py --model "ViT-B/32" --dataset "imagenet-v2" --image_emb_type "mask2former_context" --batch_size 32 --num_samples -1 > imagenet_v2_logs.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_zeroshot_cls.py --model "ViT-B/32" --dataset "imagenet-a" --image_emb_type "mask2former_context" --batch_size 32 --num_samples -1 > imagenet_a_logs.txt &
CUDA_VISIBLE_DEVICES=3 nohup python run_zeroshot_cls.py --model "ViT-B/32" --dataset "places365" --image_emb_type "mask2former_context" --batch_size 32 --num_samples -1 > places365_logs.txt &
CUDA_VISIBLE_DEVICES=4 nohup python run_zeroshot_cls.py --model "ViT-B/32" --dataset "cub" --image_emb_type "mask2former_context" --batch_size 32 --num_samples -1 > cub_logs.txt &
