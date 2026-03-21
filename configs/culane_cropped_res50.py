dataset = 'CULane_cropped'
data_root = '/media/hector/Hector/ACFR/Dataset/CULane_cropped_left'

epoch = 50
batch_size = 24            # ResNet-50 usa más VRAM que ResNet-34; bajar de 32 a 24 si hay OOM
optimizer = 'SGD'
learning_rate = 0.005
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695

# ── Backbone ──────────────────────────────────────────────────────────────────
backbone = '50'            # ResNet-50 (~25M params vs ~21M en ResNet-34)
use_aux = False
fc_norm = True

# ── Loss weights ──────────────────────────────────────────────────────────────
# sim_loss_w > 0: penaliza saltos bruscos de X entre anchors adyacentes.
#   Hace que las predicciones de carril sean más suaves y reduce FP geométricamente
#   extraños. Rango típico: 0.05–0.2. Empezar con 0.1.
sim_loss_w = 0.1

# shp_loss_w: pérdida de varianza (shape). Por defecto 0 — no activar hasta
#   tener sim_loss_w estabilizado.
shp_loss_w = 0.0

var_loss_power = 2.0

# mean_loss_w: penaliza predecir "hay carril" en anchors sin carril.
#   Aumentar de 0.05 a 0.15 reduce los FP de existencia (modelo más conservador).
#   Si el recall cae demasiado, bajar a 0.10.
mean_loss_w = 0.15

# ── Grid (igual que res34, sin cambiar para comparación justa) ────────────────
num_lanes = 2
train_width = 800
train_height = 288
num_row = 72          # 72 anchors verticales  → ~8.2 px/anchor en imagen 590px
num_col = 81          # 81 anchors horizontales → ~20.2 px/anchor en imagen 1640px
num_cell_row = 200    # 200 bins en X (1640/200 = 8.2 px/bin)
num_cell_col = 100    # 100 bins en Y (590/100 = 5.9 px/bin)
griding_num = 200     # debe ser igual a num_cell_row
crop_ratio = 1

# ── Logs y checkpoints ────────────────────────────────────────────────────────
log_path = '/media/hector/Hector/UFLDv2/logs'
note = '_culane_cropped_res50'
auto_backup = True
finetune = None
resume = None
test_model = None
test_work_dir = '/media/hector/Hector/UFLDv2/work_res50'

# ── Datos de test ─────────────────────────────────────────────────────────────
# None = list/test.txt completo (10770 imgs, ~25 min). Evaluación lenta pero fiel.
# Para evaluación rápida durante desarrollo: 'list/test_mini.txt' (2154 imgs, ~5 min)
test_list = None

# ── Datos de entrenamiento ───────────────────────────────────────────────────
# None = list/train_gt.txt (133k imgs, entrenamiento completo)
train_list = None
anno_cache = None

# ── Filtro de detecciones cortas (evaluación) ────────────────────────────────
# Fracción mínima de anchors válidos para escribir una detección al .lines.txt.
# Calibrado con el experimento res34: min_row_frac=0.7, min_col_frac=0.4
# subió F@0.5_m30 de 0.272 → 0.317 filtrando detecciones fantasma.
#
# GT real: mediana ~93% de anchors válidos. Un carril real tiene casi todos los
# anchors. Detecciones con <70% son casi siempre FP.
#
# row: de 72 anchors, requiere >50 (70%) en lugar del original >36 (50%)
# col: de 81 anchors, requiere >32 (40%) en lugar del original >20 (25%)
min_row_frac = 0.7
min_col_frac = 0.4
