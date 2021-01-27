# run tests for all different pretrained models

# EDSR-baseline x2
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=2 --model_path=models/edsr_baseline_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=2 --model_path=models/edsr_baseline_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=2 --model_path=models/edsr_baseline_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=2 --model_path=models/edsr_baseline_x2.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=2 --model_path=models/edsr_baseline_x2.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# EDSR-baseline x4
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=4 --model_path=models/edsr_baseline_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=4 --model_path=models/edsr_baseline_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=4 --model_path=models/edsr_baseline_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=4 --model_path=models/edsr_baseline_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr_baseline.json --scale=4 --model_path=models/edsr_baseline_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# EDSR x2
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# EDSR x4
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/edsr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# EUSR x2
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=2 --model_path=models/edsr_x2.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# EUSR x4
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=4 --model_path=models/edsr_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# EUSR x8
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=8 --model_path=models/edsr_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=8 --model_path=models/edsr_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=8 --model_path=models/edsr_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=8 --model_path=models/edsr_x8.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/eusr.json --scale=8 --model_path=models/edsr_x8.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# DBPN x2
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=2 --model_path=models/dbpn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=2 --model_path=models/dbpn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=2 --model_path=models/dbpn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=2 --model_path=models/dbpn_x2.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=2 --model_path=models/dbpn_x2.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# DBPN x4
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=4 --model_path=models/dbpn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=4 --model_path=models/dbpn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=4 --model_path=models/dbpn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=4 --model_path=models/dbpn_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=4 --model_path=models/dbpn_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

# DBPN x8
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=8 --model_path=models/dbpn_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=8 --model_path=models/dbpn_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=8 --model_path=models/dbpn_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=8 --model_path=models/dbpn_x8.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/dbpn.json --scale=8 --model_path=models/dbpn_x8.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG

## RCAN x2
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=2 --model_path=models/rcan_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=2 --model_path=models/rcan_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=2 --model_path=models/rcan_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=2 --model_path=models/rcan_x2.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=2 --model_path=models/rcan_x2.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## RCAN x4
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=4 --model_path=models/rcan_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=4 --model_path=models/rcan_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=4 --model_path=models/rcan_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=4 --model_path=models/rcan_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=4 --model_path=models/rcan_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## RCAN x8
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=8 --model_path=models/rcan_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=8 --model_path=models/rcan_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=8 --model_path=models/rcan_x8.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=8 --model_path=models/rcan_x8.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rcan.json --scale=8 --model_path=models/rcan_x8.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## MSRN x2
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=2 --model_path=models/msrn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=2 --model_path=models/msrn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=2 --model_path=models/msrn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=2 --model_path=models/msrn_x2.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=2 --model_path=models/msrn_x2.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## MSRN x4
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=4 --model_path=models/msrn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=4 --model_path=models/msrn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=4 --model_path=models/msrn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=4 --model_path=models/msrn_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/msrn.json --scale=4 --model_path=models/msrn_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## ESRGAN x4
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/esrgan.json --scale=4 --model_path=models/esrgan_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/esrgan.json --scale=4 --model_path=models/esrgan_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/esrgan.json --scale=4 --model_path=models/esrgan_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/esrgan.json --scale=4 --model_path=models/esrgan_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/esrgan.json --scale=4 --model_path=models/esrgan_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## RRDB x4
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rrdb.json --scale=4 --model_path=models/rrdb_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rrdb.json --scale=4 --model_path=models/rrdb_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rrdb.json --scale=4 --model_path=models/rrdb_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rrdb.json --scale=4 --model_path=models/rrdb_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/rrdb.json --scale=4 --model_path=models/rrdb_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## CARN x2
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=2 --model_path=models/carn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=2 --model_path=models/carn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=2 --model_path=models/carn_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=2 --model_path=models/carn_x2.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=2 --model_path=models/carn_x2.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## CARN x4
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=4 --model_path=models/carn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=4 --model_path=models/carn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=4 --model_path=models/carn_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=4 --model_path=models/carn_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/carn.json --scale=4 --model_path=models/carn_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## FRSR x2
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/frsr_x2.json --scale=2 --model_path=models/frsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/frsr_x2.json --scale=2 --model_path=models/frsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/frsr_x2.json --scale=2 --model_path=models/frsr_x2.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/frsr_x2.json --scale=2 --model_path=models/frsr_x2.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/frsr_x2.json --scale=2 --model_path=models/frsr_x2.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## FRSR x4
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/frsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/frsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/frsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/frsr_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/frsr_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG
#
## NatSR x4 
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/natsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_COR
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/natsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/natsr_x4.pb --data_path=/opt/temp/png_div2k/PD_2D_TRA
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/natsr_x4.pb --data_path=/opt/temp/png_div2k/T1_2D_SAG
#CUDA_VISIBLE_DEVICES=1 python get_sr.py --config_path=configs/natsr.json --scale=4 --model_path=models/natsr_x4.pb --data_path=/opt/temp/png_div2k/T1_VIBE_2D_SAG


