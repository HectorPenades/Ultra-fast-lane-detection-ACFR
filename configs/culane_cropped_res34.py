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
# Fichero de lista de test (relativo a data_root). None = usa list/test.txt completo.
# Para evaluación rápida con subconjunto aleatorio del 20%:
#   test_list = 'list/test_mini.txt'
test_list = None

# Datos de entrenamiento (relativo a data_root). None = usa list/train_gt.txt (133k imgs train).
# Para entrenar con el 80% del conjunto de test (8616 imgs, sin solapamiento con test_mini.txt):
#   train_list = 'list/train_gt_from_test.txt'
#   anno_cache = 'culane_anno_cache_test.json'
train_list = None
anno_cache = None

# Fracción mínima de anchors válidos para aceptar una detección (0.0-1.0).
# None = comportamiento original (row: >50% de 72, col: >25% de 81).
# Los GT tienen mediana ~93% de anchors. Subir min_row_frac reduce FP de detecciones cortas.
# Ejemplos:
#   min_row_frac = 0.6   → row requiere >43/72 anchors (antes >36)
#   min_col_frac = 0.35  → col requiere >28/81 anchors (antes >20, mantiene ratio similar)
min_row_frac = 0.7
min_col_frac = 0.4
