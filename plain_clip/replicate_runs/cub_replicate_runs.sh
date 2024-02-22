# Define dataset and mode arrays
datasets=("cub")
model_sizes=("ViT-B/32" "ViT-B/16" "ViT-L/14")
descriptor_paths=(
    "../descriptors/cub/sachit_descriptors_cub.json"
    "../descriptors/cub/habitat_sachit_descriptors_cub.json"
    "../descriptors/cub/chatgpt_descriptors_cub.json"
    "../descriptors/cub/habitat_chatgpt_descriptors_cub.json"
    "../descriptors/cub/ssc_descriptors_cub.json"
    "../descriptors/cub/ssch_descriptors_cub.json"
)

seeds=(
    1
    2
    3
    4
)
# Loop through each combination of dataset, mode, model size, and descriptor file
for dataset in "${datasets[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        python ../main.py --dataset="$dataset" --mode="gpt_descriptions" --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:0 --savename="normal_cub_accuracy"
    done

    for model_size in "${model_sizes[@]}"; do        
        python ../main.py --dataset="$dataset" --mode="waffle" --model_size="$model_size" --descriptor_fnames "${descriptor_paths[@]}" --device cuda:0 --savename="waffle_cub_accuracy" --seeds "${seeds[@]}"
    done
done
