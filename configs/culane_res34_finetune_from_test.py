dataset = 'CULane_cropped'
data_root = '/media/hector/Hector/ACFR/Dataset/CULane_cropped_left'

# Entrenamiento
epoch = 20
batch_size = 32

# Optimizador
optimizer = 'SGD'
learning_rate = 0.001
weight_decay = 0.0001
momentum = 0.9

# Scheduler
scheduler = 'multi'
steps = [10, 16]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100

# Modelo
backbone = '34'
use_aux = False
fc_norm = True
tta = False

# Pérdidas
sim_loss_w = 0.0
shp_loss_w = 0.0
var_loss_power = 2.0
mean_loss_w = 0.05

# Carriles
num_lanes = 2

# Resolución de entrenamiento
train_width = 800
train_height = 288

# Anchors y grillas
num_row = 72
num_col = 81
num_cell_row = 200
num_cell_col = 100
griding_num = 200

# Otros
auto_backup = True
note = '_culane_cropped_finetune_from_test'
log_path = '/media/hector/Hector/UFLDv2/logs'
finetune = '/media/hector/Hector/UFLDv2/logs/20260320_102349_lr_5e-03_b_32_culane_cropped/checkpoints/model_best.pth'
resume = None
test_model = None
test_work_dir = '/media/hector/Hector/UFLDv2/work'
crop_ratio = 1

# Listas de datos
train_list = 'list/train_gt_from_test.txt'
test_list = 'list/test_mini.txt'
