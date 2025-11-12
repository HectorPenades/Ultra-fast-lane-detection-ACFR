dataset= 'CULane'
data_root= '/home/its/Hector/dataset/InPaint/inpaint_outputs'

# Entrenamiento
epoch= 100
batch_size= 32

# Optimizador
optimizer= 'SGD'
learning_rate = 0.001
weight_decay = 0.001
momentum= 0.9

# Scheduler
scheduler= 'multi'
steps= [25,38]
gamma= 0.1
warmup= 'linear'
warmup_iters= 695

# Modelo
backbone= '34'
use_aux= False
fc_norm= True
tta=False

# Pérdidas
sim_loss_w= 0.0
shp_loss_w= 0.0
var_loss_power= 2.0
mean_loss_w= 0.05

# Carriles
num_lanes= 2

# Resolución real completa
# Final resize resolution (set to 800x288 as requested)
train_width= 800
train_height= 288

# Anchors y grillas
num_row= 72
num_col= 81
num_cell_row= 200
num_cell_col= 100
griding_num= 200 

# Otros
auto_backup= True
note= ''
log_path= ''
finetune= '/home/its/Hector/UFLDv2/Ultra-Fast-Lane-Detection-v2/20251101_135640_lr_5e-02_b_32/checkpoints/model_best.pth'
resume= None 
test_model= None
test_work_dir = '/home/its/Hector/work'
crop_ratio = 1
