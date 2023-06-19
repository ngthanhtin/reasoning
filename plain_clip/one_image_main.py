from load import *
import torchmetrics
from tqdm import tqdm
import cv2
from textwrap import wrap

seed_everything(hparams['seed'])

def draw_chart(image, predicted_classname, tensor_scores, descriptions, color='dodgerblue'):
    """
    image: np image
    predicted_classname: class name
    tensor_scores: scores of each description in descriptions [1, number_of_description]
    """
    tensor_scores = tensor_scores.squeeze()
    scores = tensor_scores.tolist()
    
    # remove which (is/has) in description
    refined_descriptions = []
    for i, des in enumerate(descriptions):
        if "which is" in des:
            index = des.find('which is')
            des = des[index + len('which is')+1:]
        elif "which has" in des:
            index = des.find('which has')
            des = des[index + len('which has')+1:]
        elif "which" in des:
            index = des.find('which')
            des = des[index + len('which')+1:]
        refined_descriptions.append(des)
    
    descriptions = refined_descriptions
    # sort scores and descriptions
    sorted_scores = sorted(scores, reverse=True)
    
    sorted_descriptions = []
    for ss in sorted_scores:
        for i, s in enumerate(scores):
            if ss == s:
                # sorted_descriptions.append(descriptions[i].split(":")[0]) # show only the visual part type
                sorted_descriptions.append(descriptions[i])
                break
    
    sorted_descriptions = [ '\n'.join(wrap(l, 50)) for l in sorted_descriptions ]
    # add average score
    avg_score = np.mean(sorted_scores)
    sorted_descriptions.insert(0, "Average Score")
    sorted_scores.insert(0, avg_score)
    
    scores = sorted_scores
    scores = [s*100 for s in scores]
    descriptions = sorted_descriptions
    
    #
    plt.rcdefaults()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    y_pos = np.arange(len(descriptions))

    ax.barh(y_pos, scores, align='center', color=color)
    # round_scores = [int(score) for score in scores]
    for i, score in enumerate(scores):
        ax.text(int(score), y_pos[i]+0.18, f'{score:.3f}', fontsize=9)
    ax.set_yticks(y_pos, labels=descriptions, fontsize=8)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('')
    ax.set_title(f'{predicted_classname}', fontsize=13)
    fig.tight_layout()

    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    pil_chart_image = Image.open(buf).convert('RGB')

    pil_bird_image = Image.fromarray(image)
    pil_bird_image.save('geed.jpg')

    def get_concat_h(im1, im2):
        height = im1.height if im1.height > im2.height else im2.height
        dst = Image.new('RGB', (im1.width + im2.width, height)) 
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    pil_image = get_concat_h(pil_bird_image, pil_chart_image)

    return pil_image

bs = hparams['batch_size']
bs = 1
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

if hparams['dataset'] == 'imagenet' or hparams['dataset'] == 'imagenetv2':
    num_classes = 1000
elif hparams['dataset'] == 'nabirds':
    num_classes = 267#555
elif hparams['dataset'] == 'cub':
    num_classes = 200
elif hparams['dataset'] == 'places365':
    num_classes = 365
elif hparams['dataset'] == 'inaturalist2021':
    num_classes = 425 #1486

# for j, (h, l) in enumerate(gpt_descriptions.items()):
#     if len(gpt_descriptions[h]) > 4:
#         print('haha')
# exit()

num_descs = 4
attributes_pc = [0 for _ in range(num_descs)]
num_correct = 0
num_habitat_correct = 0

correct_predictions = []
# f = open('abcd_nabirds.txt', 'r')
# vis_paths = f.readlines()
# vis_paths = [p[:-1] for p in vis_paths]

for batch_number, batch in enumerate(tqdm(dataloader)):
    images, labels, paths = batch
    
    images = images.to(device)
    labels = labels.to(device)
    
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)
    
    # image_labels_similarity = image_encodings @ label_encodings.T
    # clip_predictions = image_labels_similarity.argmax(dim=1)
    
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    
    for i, (k, v) in enumerate(description_encodings.items()): 
        
        
        dot_product_matrix = image_encodings @ v.T
        
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
        
        
    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        
    
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    np_image = cv2.imread(paths[0])
    
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    
    label = labels[0].detach().cpu().numpy()
    prediction = descr_predictions.data[0].item()
    
    # attributes percentage
    if prediction == label:
        correct_predictions.append(paths[0])

        tensor_scores = image_description_similarity[prediction].squeeze()
        scores = tensor_scores.tolist()
        # attributes_pc[0] += scores[0]
        # attributes_pc[1] += scores[1]
        # attributes_pc[2] += scores[2]
        # attributes_pc[3] += scores[3]
        if max(scores) == scores[-1]:
            num_habitat_correct+=1
        attributes_pc = [attributes_pc[ii] + scores[ii] for ii in range(num_descs)]
        num_correct += 1


    # convert prediction to classname
    # if prediction == label:
    #     for j, (h, l) in enumerate(gpt_descriptions.items()):
    #         if j == prediction:
    #             filename = paths[0].split('/')[-1]
    #             plot = draw_chart(np_image, h, image_description_similarity[prediction], gpt_descriptions[h])
    #             plot = plot.save(f"correct_nabirds_id_figs/{filename}.jpg")
    #             break

        # for j, (h, l) in enumerate(gpt_descriptions.items()):
        #     if j == label:
        #         plot = draw_chart(np_image, h, image_description_similarity[label], gpt_descriptions[h], color='red')
        #         plot = plot.save(f"correct_cub_id_figs/gt_{batch_number}.jpg")
        #         break

    # if paths[0] in vis_paths:    
    #     for j, (h, l) in enumerate(gpt_descriptions.items()):
    #             if j == prediction:  
    #                 plot = draw_chart(np_image, h, image_description_similarity[prediction], gpt_descriptions[h])
    #                 filename = paths[0].split('/')[-1]
    #                 plot = plot.save(f"incorrect_nabirds_id2_figs/{filename}.jpg")

                    # tensor_scores = image_description_similarity[prediction].squeeze()
                    # scores = tensor_scores.tolist()
                    # attributes_pc[0] += scores[0]
                    # attributes_pc[1] += scores[1]
                    # attributes_pc[2] += scores[2]
                    # attributes_pc[3] += scores[3]
                    # if max(scores[3], scores[2], scores[1], scores[0]) == scores[3]:
                    #     plot = draw_chart(np_image, h, image_description_similarity[prediction], gpt_descriptions[h])
                    #     filename = paths[0].split('/')[-1]
                    #     plot = plot.save(f"habitat_first_correct_nabirds_id_figs/{filename}.jpg")
                    # break

print(num_correct)
print(num_habitat_correct)
attributes_pc = [pc/num_correct for pc in attributes_pc]
print(attributes_pc)

def save_list_to_file(strings, filename):
    with open(filename, 'w') as file:
        for string in strings:
            file.write(string + '\n')

# save_list_to_file(correct_predictions, './incorrect_nabirds_id2_paths.txt')