from load import *
import torchmetrics
from tqdm import tqdm


seed_everything(hparams['seed'])

bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

print("Loading model...")

device = torch.device(hparams['device'])
# load model
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

print("Encoding descriptions...")

description_encodings = compute_description_encodings(model)

label_encodings = compute_label_encodings(model)

if hparams['dataset'] == 'cub':
    num_classes = 200
elif hparams['dataset'] == 'nabirds':
    num_classes = 267 #267
elif hparams['dataset'] == 'places365':
    num_classes = 365
elif hparams['dataset'] == 'inaturalist2021':
    num_classes = 425#1486

# load mapping
# mapping_path = 'descriptors/cub/additional_chatgpt_descriptors_cub.json'
# mapping = {}

# # read json
# with open(mapping_path, 'r') as file:
#     data = json.load(file)
# group = 0
# for i, (k, v) in enumerate(data.items()):
#     if '_A' in k:
#         mapping[group] = [i]
#     elif '_B' in k:
#         mapping[group].append(i)
#         group+=1
#     else: # standalone
#         mapping[group] = [i]
#         group+=1
##

print("Evaluating...")
lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
lang_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
clip_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

wrongly_predicted_paths = []

for batch_number, batch in enumerate(tqdm(dataloader)):
    if len(batch) == 3:
        images, labels, path = batch
    else:
        images, labels = batch
    
    images = images.to(device)
    labels = labels.to(device)
    
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    image_labels_similarity = image_encodings @ label_encodings.T
    
    # New matrix initialization
    # new_matrix = torch.zeros((image_labels_similarity.shape[0], num_classes)).to(device)
    
    # for new_index, old_indices in mapping.items():
    #     # new_matrix[:, new_index] = image_labels_similarity[:, old_indices].mean(dim=1)
    #     new_matrix[:, new_index] = image_labels_similarity[:, old_indices].max(dim=1)[0]
    #     # if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]

    # image_labels_similarity = new_matrix 
    #

    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    # Check if the prediction is incorrect
    # for i, prediction in enumerate(clip_predictions):
        # if prediction == labels[i]:
        #     wrongly_predicted_paths.append(path[i]) 

    clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
    clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)
    
    
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    
    for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
                
        dot_product_matrix = image_encodings @ v.T

        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i], aggregation_method='mean')
        

    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
    
    # New matrix initialization
    # new_matrix = torch.zeros((cumulative_tensor.shape[0], num_classes)).to(device)
    
    # for new_index, old_indices in mapping.items():
    #     # new_matrix[:, new_index] = cumulative_tensor[:, old_indices].mean(dim=1)
    #     new_matrix[:, new_index] = cumulative_tensor[:, old_indices].max(dim=1)[0]
    #     # if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]

    # cumulative_tensor = new_matrix
    
    #
    descr_predictions = cumulative_tensor.argmax(dim=1)

    lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)
    
    confmat(cumulative_tensor.softmax(dim=-1).argmax(dim=-1), labels)

# Compute the final confusion matrix
final_conf_matrix = confmat.compute()

# Calculate class-wise accuracies
class_accuracies = final_conf_matrix.diag() / final_conf_matrix.sum(1)

# Handle cases where a class never appears in the batch (to avoid division by zero)
class_accuracies[torch.isnan(class_accuracies)] = 0

# Print class-wise accuracies
for i, acc in enumerate(class_accuracies):
    print(f"Accuracy for class {i}: {acc.item() * 100:.2f}%")

# Save the accuracies to a text file
with open('class_accuracies.txt', 'w') as f:
    for i, acc in enumerate(class_accuracies):
        f.write(f"{acc.item() * 100:.2f}%\n")

# After the loop, save the paths to a text file
# with open('correctly_predicted_paths.txt', 'w') as f:
#     for path in wrongly_predicted_paths:
#         f.write(path + '\n')

print("\n")

accuracy_logs = {}
accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()
accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

# print the dictionary
print("\n")
for key, value in accuracy_logs.items():
    print(key, value)