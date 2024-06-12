python3 ../train.py --no_instance \
                    ----checkpoints_dir ../checkpoints/ \
                    --name w \
                    --dataroot ../maps/map/ \
                    --saveroot ../maps/w/ \
                    --valroot ../maps/val/ \
                    --gpu_ids 0 \
                    --save_epoch_freq 10 \
                    --loadSize 128 \
                    --splitVal \
                    --crop_and_resize \
                    --randomCrop \
                    --flip \
                    --rotate \
                    --batchSize 3 \
                    --ngf 32 \
                    --lambda_feat 5
                    # --saveImage \