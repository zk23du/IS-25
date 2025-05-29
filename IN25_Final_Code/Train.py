import torch, csv, math
import argparse, pandas
from sklearn.metrics import precision_score, recall_score, f1_score
from time import perf_counter
from torch.utils.data import DataLoader

from EMOTION_RECOGNITION import *
from EMOTION_RECOGNITION.pooling import AttentiveStatisticsPooling
from EMOTION_RECOGNITION.DataUtils import *
from EMOTION_RECOGNITION.Architectures import *


P = argparse.ArgumentParser()
P.add_argument("--config", type=str, required=True)
A = P.parse_args()

configs = GetYAMLConfigs(path=A.config)
logger = Logs(exp_name=configs["experiment_name"])
DEVICE = torch.device("cuda:{}".format(configs["cuda_device"]))

torch.cuda.empty_cache()
torch.manual_seed(configs["seed"])

############################################################
############################################################

df = pandas.read_csv(configs["label_path"])
train_df = df[df['Split_Set'] == 'Train']
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']
class_frequencies = train_df[classes].sum().to_dict()
total_samples = len(train_df)
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}

weights_list = [class_weights[cls] for cls in classes]
class_weights_tensor = torch.tensor(weights_list, device=DEVICE, dtype=torch.float)

total_dataset=dict()
total_dataloader=dict()
for dtype in ["train", "dev"]:

    cur_utts, cur_labs = load_cat_emo_label(configs["label_path"], dtype)  
    
    cur_utts = cur_utts[:100]
    cur_labs = cur_labs[:100]  
    
    cur_wavs = load_audio(configs["audio_path"], cur_utts)
    if dtype == "train":
        cur_wav_set = WavSet(cur_wavs)
        cur_wav_set.save_norm_stat(logger.chkpt_folder+"/train_norm_stat_wavlm_12.pkl")
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = load_norm_stat(logger.chkpt_folder+"/train_norm_stat_wavlm_12.pkl")
        cur_wav_set = WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)

    cur_bs = configs["BATCH_SIZE"] // configs["accumulation_steps"] if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False

    cur_emo_set = CAT_EmoSet(cur_labs)
    total_dataset[dtype] = CombinedSet([cur_wav_set, cur_emo_set, cur_utts])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=collate_fn_wav_lab_mask
    )

################
# TPL = [
#     f"encoder.layers.{i}" for i in range(9,24)
#     ] + [
#     "AttentiveStatisticsPooling",  # Train AttentiveStatisticsPooling
#     ]
################

pretrained = PreTrainedModelsSpeechBrain(name=configs["pretrained_model_name"])
# if len(TPL) != 0:
#     pretrained = PrepareArchitecture(model=pretrained, trainable_parameter_list=TPL)

pretrained = PrepareArchitecture2(model=pretrained)
pretrained.to(DEVICE)

feat_dim = pretrained.hidden_size() if callable(pretrained.hidden_size) else pretrained.hidden_size

#feat_dim = pretrained.hidden_size
#print(f"Pretrained model hidden size: {feat_dim}")  # Debug

pool_model = None
if configs["pooling_type"] and configs["pooling_type"] != "None":
    is_attentive_pooling = True
    if configs["pooling_type"] == "AttentiveStatisticsPooling":
        pool_model = AttentiveStatisticsPooling(feat_dim)  # Directly initialize
    else:
        raise ValueError(f"Unknown pooling type: {configs['pooling_type']}")
    pool_model.to(DEVICE)
else:
    is_attentive_pooling = False
    pool_model = None  # No pooling applied


concat_pool_type_list = ["AttentiveStatisticsPooling"]
dh_input_dim = feat_dim * 2 \
    if configs["pooling_type"] in concat_pool_type_list \
    else feat_dim
    
    
# trainable = DeepAudioNetEmotionClassification(
#     input_dim=dh_input_dim,          # same input dimension as before
#     conv_channels=[64, 128],         # you can adjust these as needed
#     fc_hidden=configs["head_dim"],   # using your head dimension for the FC layer
#     output_dim=8,                    # 8 output classes (emotions)
#     dropout=0.3
# )

trainable = EmotionClassification(dh_input_dim, configs["head_dim"], 1, 8, dropout=0.3)
trainable.to(DEVICE)

############################################################
# Optimizer & Scheduler
############################################################

pretrained_opt = torch.optim.AdamW(pretrained.parameters(), float(configs["learning_rate"]))
trainable_opt = torch.optim.AdamW(trainable.parameters(), float(configs["learning_rate"]))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainable_opt, T_max=configs["epochs"])

if is_attentive_pooling:
    pool_opt = torch.optim.AdamW(pool_model.parameters(), float(configs["learning_rate"]))
    pool_opt.zero_grad(set_to_none=True)
    
 
min_epoch=0
min_loss=1e10

total_correct = 0
total_samples = 0
train_preds = []
train_labels = []
dev_loss_record = [math.inf]

#criterion = FocalLoss()
for epoch in range(configs["epochs"]):
    pretrained.train()
    if pool_model:
        pool_model.train()
    trainable.train() 
    batch_cnt = 0    
    train_loss = 0

    for mbidx, xy_pair in enumerate(total_dataloader["train"]):
        x = xy_pair[0]; x=x.to(DEVICE,non_blocking=True).float()
        y = xy_pair[1]  
        if y.dim() == 2:  # If labels are one-hot encoded, convert to indices
            y = torch.argmax(y, dim=1)
        y = y.to(DEVICE, non_blocking=True).long()
        if pool_model:
            y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.to(DEVICE,non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.to(DEVICE,non_blocking=True).float()
        
        ssl = pretrained(x,attention_mask=mask)      
        #ssl = ssl.mean(dim=1)  # Apply mean pooling if AttentiveStatsPooling is None
          
        #print(f"Feature vector shape before pooling: {ssl.shape}")  # Debug
        
        if pool_model:    
            ssl = pool_model(ssl,mask)
                
        emo_pred = trainable(ssl)
        #print(f"emo_pred shape after trainable: {emo_pred.shape}")  

        loss = CE_weight_category(emo_pred, y, class_weights_tensor)  
        #loss = FocalLoss(emo_pred, y)
        total_loss = loss / configs["accumulation_steps"]
        total_loss.backward()
        
        if (batch_cnt+1) % configs['accumulation_steps'] == 0 or (batch_cnt+1) == len(total_dataloader["train"]):
            pretrained_opt.step()
            trainable_opt.step()

            pretrained.zero_grad(set_to_none=True)
            trainable.zero_grad(set_to_none=True)
        batch_cnt += 1

        pred_labels = torch.argmax(emo_pred, dim=1)
        train_preds.extend(pred_labels.cpu())
        train_labels.extend(y.cpu())
        total_correct += (pred_labels == y).sum().item()
        total_samples += y.size(0)
        train_loss += total_loss

        if mbidx%200==0:
            logger.write(
                "Epoch: [{}], MiniBatch: [{}], Loss: [{}]".format(
                    epoch, mbidx, total_loss.item()
                )
            )

    train_accuracy = total_correct / total_samples
    
    logger.write(
        "Epoch: [{}], Loss: [{}], Accuracy: [{}]".format(
            epoch, train_loss, train_accuracy
        )
    )
    
        
    pretrained.eval()
    if pool_model:
        pool_model.eval()
    trainable.eval() 
    total_pred = [] 
    total_y = []
    val_preds = []
    val_labels = []
    
    total_correct = 0
    total_samples = 0

    for mbidx, xy_pair in enumerate(total_dataloader["dev"]):
        x = xy_pair[0]; x=x.to(DEVICE,non_blocking=True).float()
        y = xy_pair[1]  
        if y.dim() == 2:  # If labels are one-hot encoded, convert to indices
            y = torch.argmax(y, dim=1)
        y = y.to(DEVICE, non_blocking=True).long()
        if pool_model:
            y = xy_pair[1]; y=y.max(dim=1)[1]; y=y.to(DEVICE,non_blocking=True).long()
        mask = xy_pair[2]; mask=mask.to(DEVICE,non_blocking=True).float()
        
        with torch.no_grad():
            ssl = pretrained(x, attention_mask=mask)
            #ssl = ssl.mean(dim=1)  # Apply mean pooling if AttentiveStatsPooling is None
            
            if pool_model:
                ssl = pool_model(ssl,mask)
                
            
            emo_pred = trainable(ssl)
    
            pred_labels = torch.argmax(emo_pred, dim=1)
            val_preds.extend(pred_labels.cpu())
            val_labels.extend(y.cpu())
            
            total_correct += (pred_labels == y).sum().item()
            total_samples += y.size(0)
            
            total_pred.append(emo_pred)
            total_y.append(y)

    dev_accuracy = total_correct / total_samples
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    loss = CE_weight_category(emo_pred, y, class_weights_tensor)
    #loss = FocalLoss(emo_pred, y)

    logger.write(
        "Epoch: [{}], DevSet Loss: [{}], DevSetAccuracy: [{}]".format(
            epoch, loss.item(), dev_accuracy
        )
    )
    
    if loss < dev_loss_record[-1]:
        dev_loss_record.append(loss.item())
        logger.write(
            "\n"+f"Saving StateDicts at epoch [{epoch}]"+"\n"
        )
        
        torch.save(
            {
                "pretrained": pretrained.state_dict(),
                "trainable": trainable.state_dict()
            },
            f=os.path.join(logger.chkpt_folder, "best_state_dict.pth")
        )
 
 
    
files_test3 = [filename for filename in os.listdir(configs["audio_path"]) if 'test3' in filename]
dtype = "test3"

total_dataset=dict()
total_dataloader=dict()

cur_wavs = load_audio(configs["audio_path"], files_test3)
wav_mean, wav_std = load_norm_stat(logger.chkpt_folder+"/train_norm_stat_wavlm_12.pkl")
cur_wav_set = WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
total_dataset[dtype] = CombinedSet([cur_wav_set, files_test3])
total_dataloader[dtype] = DataLoader(
    total_dataset[dtype], batch_size=1, shuffle=False, 
    pin_memory=True, num_workers=4,
    collate_fn=collate_fn_wav_test3
)

state_dicts = torch.load(logger.chkpt_folder+"/best_state_dict.pth", map_location=DEVICE)

pretrained.load_state_dict(state_dicts["pretrained"])
trainable.load_state_dict(state_dicts["trainable"])

pretrained.eval()
trainable.eval()

inference_time = 0
frame_sec = 0

total_correct = 0
total_samples = 0
total_pred = []
total_y = []

for dtype in ["test3"]:
    total_pred = [] 
    total_y = []
    total_utt = []
    for xy_pair in total_dataloader[dtype]:
        x = xy_pair[0]; x=x.to(DEVICE,non_blocking=True).float()
        mask = xy_pair[1]; mask=mask.to(DEVICE,non_blocking=True).float()
        fname = xy_pair[2]
        
        frame_sec += (mask.sum()/16000)
        stime = perf_counter()
        with torch.no_grad():
            ssl = pretrained(x, attention_mask=mask)
            #ssl = ssl.mean(dim=1)  # Apply mean pooling if AttentiveStatsPooling is None
            
            if pool_model:
                ssl = pool_model(ssl, mask)
                
                       
            emo_pred = trainable(ssl)

            total_pred.append(emo_pred)
            total_utt.append(fname)

        etime = perf_counter()
        inference_time += (etime-stime)

    def label_to_one_hot(label, num_classes=8):
        one_hot = ['0.0'] * num_classes
        one_hot[label.item()] = '1.0'
        return ','.join(one_hot)

    emotion_classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']

    data = []
    for pred, utt in zip(total_pred, total_utt):
        pred_values = ', '.join([f'{val:.4f}' for val in pred.cpu().numpy().flatten()])
        top_class_idx = torch.argmax(pred).item()
        top_emotion = emotion_classes[top_class_idx]
        data.append([utt[0], pred_values, top_emotion])
    
    csv_filename = os.path.join(logger.chkpt_folder, "test_results.csv")

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Prediction', 'Emotion']) 
        writer.writerows(data)

logger.write(
    "Inference Time: [{}] secs, Inference Time / second: [{}], Duration of whole test-set: [{}]".format(
        inference_time, inference_time/frame_sec, frame_sec
    )
)

