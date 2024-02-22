# Define dataset and mode arrays
datasets=("part_imagenet")
modes=(
    "gpt_descriptions" 
    "waffle"
    )
model_sizes=("ViT-B/32" "ViT-B/16" "ViT-L/14")
descriptor_paths=(
    "../descriptors/part_imagenet/78_part_imagenet_descriptions.json"
    "../descriptors/part_imagenet/78_habitat_part_imagenet_descriptions.json"
)
seeds=(
    5
    6
    7
    8
    9
    10
)

for dataset in "${datasets[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        python ../main.py --dataset="$dataset" --mode="gpt_descriptions" --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:3 --savename="normal_pi_accuracy"
    done

    for model_size in "${model_sizes[@]}"; do
        python ../main.py --dataset="$dataset" --mode="waffle" --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:3 --savename="waffle_pi_accuracy" --seeds "${seeds[@]}"
    done
done
