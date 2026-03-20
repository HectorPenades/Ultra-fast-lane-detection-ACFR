dataset = 'CULane_cropped'
data_root = '/media/hector/Hector/ACFR/Dataset/CULane_cropped_left'

epoch = 50
batch_size = 32
optimizer = 'SGD'
learning_rate = 0.005
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695

backbone = '34'
use_aux = False
fc_norm = True

sim_loss_w = 0.0
shp_loss_w = 0.0
var_loss_power = 2.0
mean_loss_w = 0.05

num_lanes = 2
train_width = 800
train_height = 288
num_row = 72          # 72 posiciones verticales, ~8px entre anchors en imagen 590px
num_col = 81          # 81 posiciones horizontales complementarias
num_cell_row = 200    # 200 bins en X (1640/200 = 8.2px/bin)
num_cell_col = 100    # 100 bins en Y (590/100 = 5.9px/bin)
griding_num = 200     # igual que num_cell_row
crop_ratio = 1        # ignorado en la versión ACFR del pipeline DALI

log_path = '/media/hector/Hector/UFLDv2/logs'
note = '_culane_cropped'
auto_backup = True
finetune = None
resume = None
test_model = None
test_work_dir = '/media/hector/Hector/UFLDv2/work'
