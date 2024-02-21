# Define dataset and mode arrays
datasets=("inaturalist")
modes=(
    "gpt_descriptions" 
    "waffle"
    )
model_sizes=("ViT-B/32" "ViT-B/16" "ViT-L/14")
descriptor_paths=(
    "../descriptors/inaturalist2021/425_sachit_descriptors_inaturalist.json"
    "../descriptors/inaturalist2021/habitat_425_sachit_descriptors_inaturalist.json"
    "../descriptors/inaturalist2021/425_chatgpt_descriptors_inaturalist.json"
    "../descriptors/inaturalist2021/habitat_425_chatgpt_descriptors_inaturalist.json"
    "../descriptors/inaturalist2021/425_ssc_descriptors_inaturalist.json"
    "../descriptors/inaturalist2021/425_ssch_descriptors_inaturalist.json"
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
        for descriptor_fname in "${descriptor_paths[@]}"; do
            # Run the Python script with the current combination
            python ../main.py --dataset="$dataset" --mode="gpt_descriptions" --model_size="$model_size" --descriptor_fname="$descriptor_fname" --device cuda:2 --savename="inat_accuracy"
        done
    done
    
    for model_size in "${model_sizes[@]}"; do
        for descriptor_fname in "${descriptor_paths[@]}"; do
            for seed in "${seeds[@]}"; do
                # Run the Python script with the current combination
                python ../main.py --dataset="$dataset" --mode="waffle" --model_size="$model_size" --descriptor_fname="$descriptor_fname" --device cuda:2 --savename="inat_accuracy" --seed=$seed
            done
        done
    done
done
